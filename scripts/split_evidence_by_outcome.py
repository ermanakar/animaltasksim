"""Split the weak-evidence history extension by previous outcome under each frozen control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

from eval.bias_history import BiasConfig, bias_features, task_design
from eval.competing_history import ComparisonConfig, comparison_design, make_folds, paired_gain, prepare_comparison
from eval.history_audit import fit_logistic
from eval.metrics import load_trials
from scripts.diagnose_bias_history import simulate

__all__ = ["Args", "VARIANTS", "PAIRS", "evaluate_split", "run"]

REWARD, ERROR = "rewarded_choice_weak", "unrewarded_choice_weak"
VARIANTS = {"base": (), "reward": (REWARD,), "error": (ERROR,), "both": (REWARD, ERROR)}
# Unique gains add one term to a model that already holds the other.
PAIRS = {"reward_alone": ("base", "reward"), "error_alone": ("base", "error"),
         "reward_unique": ("error", "both"), "error_unique": ("reward", "both"), "both": ("base", "both")}
CONTROLS = ("outcome_history", "long_history", "learning_history", "combined_history")
BIAS_CONTROLS = {"bias": "outcome_history", "bias_long": "long_history"}


@dataclass(slots=True)
class Args:
    """New output only; folds, cohort and controls come from the frozen competing-history plan."""

    previous_plan: Path = Path("runs/competing_history_20260912/plan.json")
    output: Path = Path("runs/outcome_split_20261123")
    bias: BiasConfig = field(default_factory=BiasConfig)
    recovery_seeds: tuple[int, ...] = (710, 711, 712, 713, 714)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(data: pd.DataFrame, x: np.ndarray, test: np.ndarray, control: str, fold: int,
           l2: float) -> tuple[list[pd.DataFrame], dict[str, object]]:
    """Fit the four nested variants on the other animals and score the held-out ones."""
    y = data.choice_right.to_numpy(float)
    weak, signed = data.previous_weak.to_numpy(float), data.previous_right_signed.to_numpy(float)
    success = data.previous_correct.to_numpy(float)
    terms = {REWARD: signed * weak * success, ERROR: signed * weak * (1 - success)}
    frames, weights = [], {"control": control, "fold": fold}
    for variant, extra in VARIANTS.items():
        design = np.column_stack([x] + [terms[name] for name in extra])
        beta = fit_logistic(design[~test], y[~test], l2)
        z = design[test] @ beta
        frame = data.loc[test, ["subject"]].assign(nll=np.logaddexp(0, z) - y[test] * z)
        frames.append(frame.groupby("subject", as_index=False).nll.mean().assign(control=control, model=variant))
        weights[variant] = dict(zip(extra, beta[x.shape[1]:].tolist()))
    return frames, weights


def evaluate_split(raw: pd.DataFrame, folds: list[list[str]], bias: BiasConfig) -> dict[str, object]:
    """Score every variant under every control with animal-disjoint folds."""
    frame = raw.sort_values(["session_id", "trial_index"]).reset_index(drop=True)
    if sorted(s for fold in folds for s in fold) != sorted(frame.subject.unique()):
        raise ValueError("Folds must cover animals exactly once")
    config = ComparisonConfig(l2=bias.l2)
    common = prepare_comparison(frame, config)
    designs = {control: comparison_design(common, control, config)[0] for control in CONTROLS}
    task_x = task_design(frame)[0]
    valid = frame.action.isin(["left", "right"]).to_numpy()
    y_raw = (frame.action == "right").to_numpy(float)
    additions = ["prefix_bias"] + [f"residual_trace_{rate}" for rate in bias.rates]
    frames, weights = [], []
    for fold, subjects in enumerate(folds):
        test = common.subject.isin(subjects).to_numpy()
        for control in CONTROLS:
            scored, fitted = _score(common, designs[control], test, control, fold, config.l2)
            frames += scored
            weights.append(fitted)
        # Preference controls are rebuilt per fold from the fitting animals only, as in the frozen diagnostic.
        heldout = frame.subject.isin(subjects).to_numpy()
        task_beta = fit_logistic(task_x[~heldout & valid], y_raw[~heldout & valid], bias.l2)
        data = prepare_comparison(bias_features(frame, task_beta, bias), config)
        data = data[data.after_prefix].reset_index(drop=True)
        test = data.subject.isin(subjects).to_numpy()
        for control, base in BIAS_CONTROLS.items():
            x = np.column_stack([comparison_design(data, base, config)[0], data[additions].to_numpy(float)])
            scored, fitted = _score(data, x, test, control, fold, config.l2)
            frames += scored
            weights.append(fitted)
        print(f"fold {fold + 1}/{len(folds)} scored", flush=True)
    scores = pd.concat(frames, ignore_index=True)
    gains = {control: {name: paired_gain(part, before, after) for name, (before, after) in PAIRS.items()}
             for control, part in scores.groupby("control")}
    return {"gains": gains, "weights": weights, "per_subject": scores.to_dict("records"),
            "subjects": int(scores.subject.nunique())}


def run(args: Args) -> None:
    """Record the prediction before fitting, then score real animals and synthetic generators."""
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a fresh output directory")
    previous = json.loads(args.previous_plan.read_text())
    if _digest(args.previous_plan) != args.previous_plan.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Previous plan hash mismatch")
    if any(_digest(Path(p)) != value for p, value in previous["input_sha256"].items()):
        raise RuntimeError("Previous inputs changed")
    identities = {r["id"]: r for r in json.loads(Path(previous["settings"]["identities"]).read_text())["sessions"]}
    raw = load_trials(Path(previous["settings"]["reference"]))
    for name in ("subject", "lab"):
        raw[name] = raw.session_id.map(lambda sid: identities[sid][name])
    if raw.subject.isin(previous["reserved_subjects_excluded"]).any():
        raise RuntimeError("Reserved animals overlap")
    regimes = ["static_bias", "slow_drift", "history_only", "mixed"]
    plan = {"status": "exploratory_previously_inspected_development_cohort",
            "previous_plan_sha256": _digest(args.previous_plan), "script_sha256": _digest(Path(__file__)),
            "bias": asdict(args.bias), "folds": previous["subject_folds"], "variants": VARIANTS, "pairs": PAIRS,
            "controls": list(CONTROLS) + list(BIAS_CONTROLS),
            "question": "Is the weak-evidence history gain carried by the term after rewards, after errors, or both?",
            "prediction": "If the lead from the deep exploration holds, reward_unique shrinks to an interval "
                          "containing zero under bias_long while error_unique stays positive.",
            "recovery": {"regimes": regimes, "seeds": list(args.recovery_seeds), "folds": 3,
                         "truth": "history_only and mixed contain a real error term and no reward term; "
                                  "static_bias and slow_drift contain neither."},
            "limits": ["Previously inspected animals; five correlated contrasts per control; descriptive intervals.",
                       "The following-trial control from the exploration is not used here; controls are causal only.",
                       "Recovery uses fixed illustrative strengths, not a power study."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = evaluate_split(raw, plan["folds"], args.bias)
    pd.DataFrame(results.pop("per_subject")).to_csv(args.output / "subject_scores.csv", index=False)
    (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    recovery = []
    for regime in regimes:
        for seed in args.recovery_seeds:
            simulated = simulate(regime, seed)
            summary = evaluate_split(simulated, make_folds(simulated, ComparisonConfig(folds=3)), args.bias)
            recovery.append({"regime": regime, "seed": seed, "gains": summary["gains"]})
    (args.output / "recovery.json").write_text(json.dumps(recovery, indent=2, allow_nan=False) + "\n")
    print(json.dumps({c: {n: [round(g["mean_gain"], 6)] + [round(v, 6) for v in g["subject_bootstrap_95"]]
                          for n, g in pairs.items()} for c, pairs in results["gains"].items()}, indent=1))


if __name__ == "__main__":
    run(tyro.cli(Args))
