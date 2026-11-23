"""Exploratory descriptive summaries of the deep development reference; no claims, no confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

__all__ = ["Args", "load", "fit_history", "run"]

NEED = ("choice", "contrastLeft", "contrastRight", "feedbackType", "probabilityLeft", "response_times", "stimOn_times")
HISTORY = ("reward_hard", "reward_easy", "error_hard", "error_easy")


@dataclass(slots=True)
class Args:
    """Reads verified raw arrays; writes one summary JSON to a new directory."""

    source: Path = Path("runs/ibl_deep_reference_v1")
    output: Path = Path("runs/ibl_deep_explore_20261123")
    seed: int = 20261123


def load(source: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """One row per trial. ALF choice -1 is a rightward choice (checked against feedback below)."""
    frames, skipped = [], []
    for s in json.loads((source / "source_manifest.json").read_text())["sessions"]:
        with np.load(source / "raw" / f"{s['id']}.npz") as a:
            n = len(a["choice"]) if "choice" in a.files else 0
            if n == 0 or any(k not in a.files or len(a[k]) != n for k in NEED):
                skipped.append({"id": s["id"], "family": s["family"]})
                continue
            left, right = a["contrastLeft"].astype(float), a["contrastRight"].astype(float)
            frame = pd.DataFrame({
                "choice": a["choice"].astype(int), "signed": np.nan_to_num(right) - np.nan_to_num(left),
                "side": np.where(np.isnan(left), 1, -1), "fb": a["feedbackType"].astype(int),
                "pl": np.round(a["probabilityLeft"].astype(float), 2),
                "rt": a["response_times"] - a["stimOn_times"], "trial": np.arange(n)})
        for key in ("subject", "lab", "family"):
            frame[key] = s[key]
        frame["eid"], frame["start"] = s["id"], s["start_time"]
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    committed = df.choice != 0
    agree = ((df.choice == -df.side) == (df.fb == 1))[committed]
    if agree.mean() < 0.999:
        raise ValueError("Choice coding disagrees with feedback.")
    df["right"] = (df.choice == -1).astype(float)
    df["strength"] = df.signed.abs().round(4)
    order = df.groupby("subject").start.transform(lambda x: pd.factorize(np.sort(x.unique()))[0][np.searchsorted(np.sort(x.unique()), x)])
    df["session_number"] = order
    # Previous trial in the same session; history is undefined across omissions.
    same = (df.eid == df.eid.shift()) & (df.choice.shift() != 0) & committed
    df["prev_choice"] = np.where(same, -df.choice.shift(), np.nan)  # +1 right
    df["prev_reward"] = np.where(same, df.fb.shift() == 1, np.nan)
    df["prev_strength"] = np.where(same, df.strength.shift(), np.nan)
    # The following trial cannot cause this choice; it is the control for slow side biases.
    ahead = (df.eid == df.eid.shift(-1)) & (df.choice.shift(-1) != 0) & committed
    df["next_choice"] = np.where(ahead, -df.choice.shift(-1), np.nan)
    df["next_reward"] = np.where(ahead, df.fb.shift(-1) == 1, np.nan)
    df["next_strength"] = np.where(ahead, df.strength.shift(-1), np.nan)
    return df, skipped


def _logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        step = np.linalg.solve((x * (p * (1 - p))[:, None]).T @ x + ridge * np.eye(len(w)), x.T @ (y - p) - ridge * w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit_history(data: pd.DataFrame, other: str = "prev", minimum: int = 60) -> dict[str, float] | None:
    """Choice ~ one weight per signed contrast + block + neighbouring choice split by its outcome and difficulty.

    `other="prev"` uses the previous trial; `other="next"` uses the following trial as an acausal control.
    """
    d = data[data[f"{other}_choice"].notna()]
    hard = d[f"{other}_strength"] <= 0.125
    reward = d[f"{other}_reward"] == 1
    cells = {"reward_hard": reward & hard, "reward_easy": reward & ~hard,
             "error_hard": ~reward & hard, "error_easy": ~reward & ~hard}
    if min(int(m.sum()) for m in cells.values()) < minimum:
        return None
    levels = np.sort(d.signed.unique())
    x = [(d.signed.to_numpy() == level).astype(float) for level in levels]
    x.append(np.select([d.pl == 0.2, d.pl == 0.8], [1.0, -1.0], 0.0))
    x += [d[f"{other}_choice"].to_numpy() * cells[name].to_numpy() for name in HISTORY]
    w = _logistic(np.column_stack(x), d.right.to_numpy())
    return dict(zip(HISTORY, w[-4:].tolist())) | {"block": float(w[-5]), "trials": len(d)}


def _interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Equal-animal mean with a 95% bootstrap interval over animals."""
    values = values[np.isfinite(values)]
    boot = rng.choice(values, (2000, len(values))).mean(axis=1)
    return [float(values.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), len(values)]


def run(args: Args) -> None:
    """Compute every summary as an equal-animal average."""
    path = args.output / "summary.json"
    if path.exists():
        raise ValueError("Summary exists; use a new output directory.")
    rng = np.random.default_rng(args.seed)
    df, skipped = load(args.source)
    done = df[df.choice != 0]
    blocks = df[df.family.isin(["biased", "ephys"])]
    out: dict[str, object] = {"skipped_sessions": skipped, "overview": {
        "trials": len(df), "sessions": int(df.eid.nunique()), "subjects": int(df.subject.nunique()),
        "labs": int(df.lab.nunique()), "by_family": df.groupby("family").eid.nunique().to_dict(),
        "trials_by_family": df.groupby("family").size().to_dict(),
        "sessions_per_subject": sorted(df.groupby("subject").eid.nunique().tolist()),
        "omission_rate": float((df.choice == 0).mean())}}

    # 1. Learning: accuracy on full-contrast trials, sessions aligned to the first block session.
    first = blocks.groupby("subject").session_number.min()
    easy = done[done.strength == 1].assign(ok=lambda d: d.fb == 1)
    per = easy.groupby(["subject", "session_number"]).ok.mean().reset_index()
    per["rel"] = per.session_number - per.subject.map(first)
    wide = per[per.rel.between(-30, 40)].pivot(index="subject", columns="rel", values="ok")
    out["learning"] = [{"rel": int(r), "median": float(wide[r].median()), "q1": float(wide[r].quantile(.25)),
                        "q3": float(wide[r].quantile(.75)), "n": int(wide[r].notna().sum())}
                       for r in wide.columns if wide[r].notna().sum() >= 20]
    out["sessions_before_blocks"] = sorted(int(v) for v in first)

    # 2. Psychometric curves by block.
    psych = done[done.family.isin(["biased", "ephys"])].groupby(["subject", "pl", "signed"]).right.mean().reset_index()
    out["psychometric"] = [{"pl": float(pl), "signed": float(c), "mean": _interval(g.right.to_numpy(), rng)}
                           for (pl, c), g in psych.groupby(["pl", "signed"])]

    # 3. Block switches: choices toward the new block's side on weak stimuli.
    b = blocks.copy()
    switch = (b.eid == b.eid.shift()) & (b.pl != b.pl.shift()) & (b.pl.shift() != 0.5)
    b["block_id"] = (switch | (b.eid != b.eid.shift())).cumsum()
    b["since"] = b.groupby("block_id").cumcount()
    b["switched"] = b.groupby("block_id").trial.transform("first") != b.groupby("eid").trial.transform("first")
    b["until"] = b.since - b.groupby("block_id").since.transform("size")
    b["favoured"] = np.where(b.pl == 0.2, 1.0, 0.0)
    weak = b[(b.strength <= 0.0625) & (b.choice != 0) & (b.pl != 0.5)]
    after = weak[weak.switched & (weak.since < 60)].assign(rel=lambda d: d.since, toward=lambda d: d.right == d.favoured)
    # Before a switch, score choices toward the side the *next* block will favour.
    before = weak[weak.until >= -20].assign(rel=lambda d: d.until, toward=lambda d: d.right != d.favoured)
    both = pd.concat([before, after])
    both["bin"] = (both.rel // 5) * 5
    curve = both.groupby(["subject", "bin"]).toward.mean().reset_index()
    out["block_switch"] = [{"rel": int(k), "mean": _interval(g.toward.to_numpy(float), rng)} for k, g in curve.groupby("bin")]
    out["block_lengths"] = {"median": float(b[b.pl != 0.5].groupby("block_id").size().median())}

    # 4. History weights by context, per animal.
    full = done.groupby("eid").strength.transform("min") == 0
    contexts = {"training": done[(done.family == "training") & (done.pl == 0.5) & full],
                "unbiased_start": done[done.family.isin(["biased", "ephys"]) & (done.pl == 0.5)],
                "blocks": done[done.family.isin(["biased", "ephys"]) & (done.pl != 0.5)]}
    history: dict[str, object] = {}
    for name, data in contexts.items():
        fits = {s: (p, f) for s, g in data.groupby("subject")
                if (p := fit_history(g)) and (f := fit_history(g, "next"))}
        past = pd.DataFrame({s: v[0] for s, v in fits.items()}).T
        future = pd.DataFrame({s: v[1] for s, v in fits.items()}).T
        net = past[list(HISTORY)] - future[list(HISTORY)]
        history[name] = {"past": {k: _interval(past[k].to_numpy(float), rng) for k in HISTORY},
                         "future": {k: _interval(future[k].to_numpy(float), rng) for k in HISTORY},
                         "net": {k: _interval(net[k].to_numpy(float), rng) for k in HISTORY},
                         "block": _interval(past.block.to_numpy(float), rng),
                         "net_difficulty_after_reward": _interval((net.reward_hard - net.reward_easy).to_numpy(float), rng),
                         "animals_net_difficulty_positive": int((net.reward_hard > net.reward_easy).sum()),
                         "raw_difficulty_after_reward": _interval((past.reward_hard - past.reward_easy).to_numpy(float), rng),
                         "animals": len(past), "trials": int(past.trials.sum())}
    out["history"] = history

    # 5. Is the history habit a stable trait? Odd versus even block sessions of each animal.
    data = contexts["blocks"]
    parity = data.groupby("subject").session_number.transform(lambda x: pd.factorize(x, sort=True)[0] % 2)
    halves = []
    for subject, g in data.groupby("subject"):
        row = {"subject": subject, "lab": g.lab.iloc[0], "sessions": int(g.eid.nunique())}
        for half in (0, 1):
            part = g[parity[g.index] == half]
            p, f = fit_history(part), fit_history(part, "next")
            if not (p and f):
                break
            # Side stickiness: what past and future share. Outcome effect: what only the past carries.
            row[f"sticky_{half}"] = float(np.mean([f[k] for k in HISTORY]))
            row[f"win_{half}"] = float(np.mean([p[k] - f[k] for k in HISTORY[:2]]))
            row[f"lose_{half}"] = float(np.mean([p[k] - f[k] for k in HISTORY[2:]]))
        else:
            halves.append(row)
    table = pd.DataFrame(halves)
    out["trait"] = {"animals": halves} | {f"{k}_r": float(np.corrcoef(table[f"{k}_0"], table[f"{k}_1"])[0, 1])
                                          for k in ("sticky", "win", "lose")}

    # 6. Engagement within block sessions.
    e = blocks.assign(bin=(blocks.trial // 100) * 100)
    e = e[e.bin < 1200]
    rows = []
    for k, g in e.groupby("bin"):
        committed = g[g.choice != 0]
        strong = committed[committed.strength == 1]
        late = {}
        for label, part in (("win", committed[committed.prev_reward == 1]), ("lose", committed[committed.prev_reward == 0])):
            zero = part[part.strength <= 0.0625]
            late[label] = _interval(zero.assign(stay=(zero.right * 2 - 1) == zero.prev_choice).groupby("subject").stay.mean().to_numpy(float), rng)
        rows.append({"bin": int(k), "sessions": int(g.eid.nunique()),
                     "easy_accuracy": _interval(strong.assign(ok=strong.fb == 1).groupby("subject").ok.mean().to_numpy(float), rng),
                     "median_rt": _interval(committed.groupby("subject").rt.median().to_numpy(float), rng),
                     "omissions": _interval(g.assign(o=g.choice == 0).groupby("subject").o.mean().to_numpy(float), rng),
                     "stay_after_win": late["win"], "stay_after_loss": late["lose"]})
    out["engagement"] = rows
    out["session_length"] = {"median": float(blocks.groupby("eid").size().median())}

    args.output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: out[k] for k in ("history", "block_lengths", "session_length")}))
    print("trait", {k: v for k, v in out["trait"].items() if k != "animals"}, len(halves))


if __name__ == "__main__":
    run(tyro.cli(Args))
