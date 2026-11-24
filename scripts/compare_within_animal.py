"""Within-animal comparison of fixed, history, drift and switching models on the deep development reference."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from eval.within_animal import MODELS, Sessions, design, fit, pad, predict
from eval.within_animal_recovery import CONTRASTS_TESTED, DELTA, _subset, decide

__all__ = ["Args", "DESCRIPTIVE", "JOBS", "load_sessions", "run", "score", "summarize"]

CODE = (Path("scripts/compare_within_animal.py"), Path("eval/within_animal.py"), Path("eval/within_animal_recovery.py"))
PROTOCOL = Path("docs/WITHIN_ANIMAL_COMPARISON_PROTOCOL.md")
DRIFT = ("drift", "drift_history")
# job name -> (split, drift states, models)
JOBS = {"alternating": ("alternating", 41, MODELS), "chronological": ("chronological", 41, MODELS),
        "alternating_states81": ("alternating", 81, DRIFT)}
DESCRIPTIVE = {"history_total": ("fixed", "history"), "drift_total": ("fixed", "drift"), "switching_total": ("fixed", "switching")}
SHARES = {"history_share_after_drift": ("history_drift", "history_total"),
          "history_share_after_switching": ("history_switching", "history_total")}


@dataclass(slots=True)
class Args:
    """`plan` freezes inputs, code and protocol; `run` fits and prints no score; `summarize` applies the rules once."""

    stage: Literal["plan", "run", "summarize"]
    converted: Path = Path("runs/ibl_deep_converted_v1")
    output: Path = Path("runs/within_animal_comparison_v1")
    min_trials: int = 150
    min_sessions: int = 6
    maxiter: int = 5000
    resamples: int = 10000
    workers: int = 8


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sessions(converted: Path, min_trials: int, min_sessions: int) -> dict[str, tuple[str, Sessions]]:
    """Lab and padded trials per animal, in session order, with real choices and outcomes."""
    manifest = json.loads((converted / "manifest.json").read_text())
    eligible = {s["id"]: s for s in manifest["sessions"]
                if s["status"] == "converted" and s["family"] == "biased" and s["trials"] >= min_trials}
    animals = {}
    for name, record in sorted(manifest["files"].items()):
        if not name.startswith("biased/"):
            continue
        if _digest(converted / name) != record["sha256"]:
            raise RuntimeError(f"Converted file changed: {name}")
        rows: dict[str, list[tuple[float, float, float, float]]] = {}
        with (converted / name).open() as lines:
            for line in lines:
                trial = json.loads(line)
                if trial["session_id"] in eligible:
                    # Schema: action 0 is right, 1 is left, 2 is an omission. Models use +1 right, -1 left, 0 omitted.
                    choice = {0: 1.0, 1: -1.0, 2: 0.0}[trial["action"]]
                    rows.setdefault(trial["session_id"], []).append(
                        (trial["stimulus"]["contrast"], trial["block_prior"]["p_right"], choice, float(bool(trial["correct"]) and choice != 0)))
        if len(rows) < min_sessions:
            continue
        ordered = sorted(rows, key=lambda eid: eligible[eid]["session_number"])
        values = [np.asarray(rows[eid]) for eid in ordered]
        for block in values:  # the sign convention is checked on structure: a correct choice has the sign of a non-zero contrast
            informative = (block[:, 0] != 0) & (block[:, 2] != 0)
            if not np.array_equal(block[informative, 3] == 1, np.sign(block[informative, 0]) == block[informative, 2]):
                raise RuntimeError(f"Choice sign and outcome disagree in {name}")
        animals[Path(name).stem] = (eligible[ordered[0]]["lab"], pad(
            [{"contrast": v[:, 0], "prior": v[:, 1], "choice": v[:, 2], "correct": v[:, 3]} for v in values]))
    return animals


def score(data: Sessions, split: str, states: int, models: tuple[str, ...], maxiter: int) -> dict[str, object]:
    """Cross-fitted held-out loss per committed trial; halves alternate by session or divide the sessions in time."""
    full, count = design(data), len(data.length)
    if split == "alternating":
        halves = [np.arange(0, count, 2), np.arange(1, count, 2)]
    elif split == "chronological":
        halves = [np.arange(0, (count + 1) // 2), np.arange((count + 1) // 2, count)]
    else:
        raise ValueError(f"Unknown split {split}")
    committed = int(full.committed.sum())
    out: dict[str, object] = {"sessions": count, "trials": committed, "nll": {}, "converged": {}, "drift": {}}
    for model in models:
        total, converged, parameters = 0.0, [], []
        for train, test in (halves, halves[::-1]):
            fitted = fit(_subset(full, train, data.length), model, maxiter=maxiter, states=states)
            total += float(np.nansum(predict(_subset(full, test, data.length), fitted)))
            converged.append(fitted.converged)
            parameters.append(fitted.extra)
        out["nll"][model] = total / committed
        out["converged"][model] = bool(all(converged))
        if model in DRIFT:
            out["drift"][model] = {key: float(np.mean([p[key] for p in parameters])) for key in ("rho", "sd")}
    return out


def _plan(args: Args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    animals = load_sessions(args.converted, args.min_trials, args.min_sessions)
    plan = {"status": "exploratory: development animals; first fit of within-animal models to real choices",
            "protocol": str(PROTOCOL), "protocol_sha256": _digest(PROTOCOL), "models": list(MODELS), "delta": DELTA,
            "contrasts": CONTRASTS_TESTED, "descriptive": DESCRIPTIVE, "maxiter": args.maxiter, "resamples": args.resamples,
            "jobs": {name: [split, states, list(models)] for name, (split, states, models) in JOBS.items()},
            "eligibility": {"family": "biased", "min_trials": args.min_trials, "min_sessions": args.min_sessions},
            "animals": {name: {"lab": lab, "sessions": len(s.length), "trials": int(s.length.sum())}
                        for name, (lab, s) in animals.items()},
            "converted_manifest_sha256": _digest(args.converted / "manifest.json"),
            "code_sha256": {str(p): _digest(p) for p in CODE}}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "plan.json"
    path.write_text(json.dumps(plan, indent=1) + "\n")
    path.with_suffix(".sha256").write_text(_digest(path) + "\n")
    print(json.dumps({"animals": len(animals), "sessions": sum(a["sessions"] for a in plan["animals"].values()),
                      "trials": sum(a["trials"] for a in plan["animals"].values())}))


def _frozen(args: Args) -> dict:
    path = args.output / "plan.json"
    if _digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Plan changed after freezing")
    plan = json.loads(path.read_text())
    if any(_digest(Path(p)) != value for p, value in plan["code_sha256"].items()):
        raise RuntimeError("Code changed after the plan was frozen")
    if _digest(PROTOCOL) != plan["protocol_sha256"]:
        raise RuntimeError("Protocol changed after the plan was frozen")
    return plan


def _work(task: tuple[str, Sessions, str, int, tuple[str, ...], int]) -> dict[str, object]:
    name, data, split, states, models, maxiter = task
    return {"subject": name} | score(data, split, states, models, maxiter)


def _run(args: Args) -> None:
    plan = _frozen(args)
    animals = load_sessions(args.converted, args.min_trials, args.min_sessions)
    if {n: {"lab": lab, "sessions": len(s.length), "trials": int(s.length.sum())} for n, (lab, s) in animals.items()} != plan["animals"]:
        raise RuntimeError("Sessions differ from the frozen plan")
    scores = args.output / "scores"
    scores.mkdir(exist_ok=True)
    pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    try:
        for name, (split, states, models) in plan["jobs"].items():
            path = scores / f"{name}.json"
            if path.exists():
                continue
            tasks = [(subject, data, split, states, tuple(models), plan["maxiter"]) for subject, (_, data) in animals.items()]
            done = list(pool.map(_work, tasks, chunksize=1) if pool else map(_work, tasks))
            partial = path.with_suffix(".partial")
            partial.write_text(json.dumps({"job": name, "animals": done}, indent=1) + "\n")
            partial.replace(path)
            print(f"finished {name}", flush=True)  # deliberately no score
    finally:
        if pool:
            pool.shutdown()


def _table(animals: list[dict], pairs: dict[str, tuple[str, str]]) -> np.ndarray:
    return np.array([[a["nll"][before] - a["nll"][after] for before, after in pairs.values()] for a in animals])


def _endpoints(animals: list[dict], rng: np.random.Generator, resamples: int) -> dict[str, object]:
    """Decisions under the frozen rules, with every mean gain, interval and count of animals with a positive gain."""
    pairs = CONTRASTS_TESTED | DESCRIPTIVE
    gains = _table(animals, pairs)
    index = rng.integers(len(gains), size=(resamples, len(gains)))
    means = gains[index].mean(1)
    low, high = np.quantile(means, [0.025, 0.975], axis=0)
    column = {name: i for i, name in enumerate(pairs)}
    summary = {name: {"mean": float(gains[:, i].mean()), "interval": [float(low[i]), float(high[i])],
                      "animals_positive": int((gains[:, i] > 0).sum())} for name, i in column.items()}
    shares = {}
    for name, (top, bottom) in SHARES.items():
        ratio = means[:, column[top]] / means[:, column[bottom]]
        shares[name] = {"ratio_of_means": float(gains[:, column[top]].mean() / gains[:, column[bottom]].mean()),
                        "interval": [float(v) for v in np.quantile(ratio, [0.025, 0.975])]}
    return {"animals": len(animals), "decisions": decide(gains[:, :len(CONTRASTS_TESTED)], rng, resamples),
            "gains": summary, "shares": shares}


def summarize(plan: dict, jobs: dict[str, list[dict]], seed: int = 20261124) -> dict:
    """Run conditions, primary endpoints, the two planned sensitivity analyses and descriptive tables."""
    rng = np.random.default_rng(seed)
    primary, fine = jobs["alternating"], jobs["alternating_states81"]
    labs = {name: record["lab"] for name, record in plan["animals"].items()}
    largest = max(sorted(set(labs.values())), key=list(labs.values()).count)
    nonconverged = float(np.mean([not ok for a in primary for ok in a["converged"].values()]))
    resolution = float(np.mean([max(abs(a["nll"][m] - b["nll"][m]) for m in DRIFT) for a, b in zip(primary, fine, strict=True)]))
    conditions = {"nonconverged_fraction": nonconverged, "numerical_resolution": resolution,
                  "met": bool(nonconverged <= 0.02 and resolution <= 2e-4)}
    main = _endpoints(primary, rng, plan["resamples"])
    sensitivity = {"chronological": _endpoints(jobs["chronological"], rng, plan["resamples"]),
                   "without_largest_lab": {"lab_removed": largest} | _endpoints(
                       [a for a in primary if labs[a["subject"]] != largest], rng, plan["resamples"])}
    robust = {key: all(s["decisions"][key] == main["decisions"][key] for s in sensitivity.values())
              for key in ("state", "history", "form")}
    best = [min(a["nll"], key=a["nll"].get) for a in primary]
    gains = _table(primary, CONTRASTS_TESTED | DESCRIPTIVE)
    by_lab = {lab: {"animals": sum(v == lab for v in labs.values())} | {
        name: float(gains[[labs[a["subject"]] == lab for a in primary], i].mean())
        for i, name in enumerate(CONTRASTS_TESTED | DESCRIPTIVE)} for lab in sorted(set(labs.values()))}
    return {"status": plan["status"], "conditions": conditions,
            "decisions": main["decisions"] if conditions["met"] else None, "robust": robust if conditions["met"] else None,
            "primary": main, "sensitivity": sensitivity,
            "mean_nll": {model: float(np.mean([a["nll"][model] for a in primary])) for model in MODELS},
            "best_model_counts": {model: best.count(model) for model in MODELS},
            "nonconverged_by_model": {model: sum(not a["converged"][model] for a in primary) for model in MODELS},
            "fitted_drift": {model: {key: [float(v) for v in np.quantile([a["drift"][model][key] for a in primary], [0.25, 0.5, 0.75])]
                                     for key in ("rho", "sd")} for model in DRIFT},
            "by_lab": by_lab}


def _summarize(args: Args) -> None:
    path = args.output / "result.json"
    if path.exists():
        raise RuntimeError("Result exists; it is never replaced")
    plan = _frozen(args)
    jobs = {name: json.loads((args.output / "scores" / f"{name}.json").read_text())["animals"] for name in plan["jobs"]}
    result = summarize(plan, jobs)
    result["plan_sha256"] = _digest(args.output / "plan.json")
    with (args.output / "subject_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "lab", "split", "sessions", "trials", *MODELS])
        for split in ("alternating", "chronological"):
            for a in jobs[split]:
                writer.writerow([a["subject"], plan["animals"][a["subject"]]["lab"], split, a["sessions"], a["trials"],
                                 *(f"{a['nll'][m]:.6f}" for m in MODELS)])
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("conditions", "decisions", "robust", "primary", "mean_nll", "best_model_counts")}, indent=1))


def run(args: Args) -> None:
    """Dispatch one explicit stage."""
    {"plan": _plan, "run": _run, "summarize": _summarize}[args.stage](args)


if __name__ == "__main__":
    run(tyro.cli(Args))
