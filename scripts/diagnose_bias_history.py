"""Freeze causal bias controls and run development and synthetic diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from scipy.special import expit

from eval.bias_history import BiasConfig, evaluate_bias
from eval.competing_history import ComparisonConfig, make_folds, prepare_comparison
from eval.metrics import load_trials


@dataclass(slots=True)
class Args:
    """A new experiment, preserving earlier plans and results."""

    previous_plan: Path = Path("runs/competing_history_20260912/plan.json")
    output: Path = Path("runs/bias_history_20260912")
    config: BiasConfig = field(default_factory=BiasConfig)


def digest(path: Path) -> str:
    """Hash immutable study inputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate(regime: str, seed: int) -> pd.DataFrame:
    """Generate explicit static/drifting preference and genuine-history alternatives."""
    if regime not in ("static_bias", "slow_drift", "history_only", "mixed"):
        raise ValueError("Unknown generator")
    rng = np.random.default_rng(seed)
    rows = []
    for animal in range(12):
        bias = rng.normal(0, .9) if regime != "history_only" else 0.
        drift = 0.
        actions, successes, strengths = [], [], []
        for trial in range(500):
            contrast = rng.choice([-.25, -.125, -.0625, .0625, .125, .25])
            if regime in ("slow_drift", "mixed"):
                drift = .99 * drift + rng.normal(0, .15)
            score = 4 * contrast + bias + drift
            if regime in ("history_only", "mixed"):
                if trial >= 3:
                    score += 1.2 * actions[-3] * successes[-3]
                if trial:
                    score += .4 * actions[-1] * (1-successes[-1]) * (strengths[-1] <= .125)
            action = 1 if rng.random() < expit(score) else -1
            correct = action == np.sign(contrast)
            rows.append({"task": "ibl_2afc", "subject": f"m{animal}", "lab": f"lab{animal%3}",
                         "session_id": f"s{animal}", "trial_index": trial, "stimulus_contrast": contrast,
                         "action": "right" if action == 1 else "left", "correct": bool(correct),
                         "block_prior": {"p_right": .5}})
            actions.append(action)
            successes.append(int(correct))
            strengths.append(abs(contrast))
    return pd.DataFrame(rows)


def run(args: Args) -> None:
    """Freeze rules before real fits; retain all synthetic repetitions regardless of outcome."""
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a fresh output directory")
    previous = json.loads(args.previous_plan.read_text())
    if digest(args.previous_plan) != args.previous_plan.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Previous plan hash mismatch")
    for p, value in previous["input_sha256"].items():
        if digest(Path(p)) != value:
            raise RuntimeError("Previous inputs changed")
    reference = Path(previous["settings"]["reference"])
    identities = {r["id"]: r for r in json.loads(Path(previous["settings"]["identities"]).read_text())["sessions"]}
    raw = load_trials(reference).sort_values(["session_id", "trial_index"]).reset_index(drop=True)
    for name in ("subject", "lab"):
        raw[name] = raw.session_id.map(lambda sid: identities[sid][name])
    if raw.subject.isin(previous["reserved_subjects_excluded"]).any():
        raise RuntimeError("Reserved animals overlap")
    raw["ordinal"] = raw.groupby("session_id").cumcount()
    prefix = raw[raw.ordinal < args.config.prefix]
    counts = prefix.assign(valid=prefix.action.isin(["left", "right"])).groupby("session_id").valid.sum()
    common = prepare_comparison(raw, ComparisonConfig())
    common = common[(common.ordinal >= args.config.prefix) &
                    (common.session_id.map(counts) >= args.config.min_prefix_choices)]
    inputs = {**previous["input_sha256"], str(args.previous_plan): digest(args.previous_plan)}
    paths = [Path("scripts/diagnose_bias_history.py"), Path("eval/bias_history.py"),
             Path("eval/competing_history.py"), Path("eval/history_audit.py"), Path("eval/metrics.py")]
    code = {str(p): digest(p) for p in paths}
    plan = {"config": asdict(args.config), "input_sha256": inputs, "source_sha256": code,
            "folds": previous["subject_folds"], "scored_trials": len(common),
            "subjects": common.subject.nunique(), "sessions": common.session_id.nunique(),
            "status": "exploratory_development_diagnostic", "reserved_subjects_excluded": previous["reserved_subjects_excluded"],
            "primary": "bias_long_evidence vs bias_long; positive equal-animal NLL reduction favors added evidence.",
            "secondary": "Added lags after bias controls; bias and long history over one-step history; no best-model selection.",
            "bias": "Task-only ridge logistic fitted on fitting animals; prefix-only ridge1 scalar bias bounded[-4,4] uses first100rawtrials/min80choices. All models score only later common rows.",
            "slow_state": "Causal EWMA of y minus task-only probability at fixed rates .005,.02. Reset on session/gap/omission; update only after prediction. Not an actual PsyTrack/GLM-HMM fit.",
            "recovery": {"regimes": ["static_bias", "slow_drift", "history_only", "mixed"],
                         "seeds": [710,711,712,713,714], "animals": 12, "trials_per_animal": 500,
                         "folds": 3, "seed": 20260912,
                         "decision": "Report all means, paired intervals and positive-lower-bound flags. No tuning or selection. Five repetitions and fixed synthetic strengths are illustrative, not a power study or calibrated false-positive guarantee."},
            "limits": ["Bias summaries may also absorb real memory; gains cannot identify a mechanism.",
                       "Analyst block controls are available to all models; no autonomous-agent claim.",
                       "Previously inspected animals, overlapping fitted folds, descriptive bootstrap, multiple contrasts.",
                       "No cross-lab confirmation here; prefix restriction changes the sample.",
                       "Simple diagnostic controls do not replace published generative state-model comparisons.",
                       "Do not access a fresh confirmation cohort on the basis of these diagnostics alone."]}
    args.output.mkdir(parents=True, exist_ok=True)
    p = args.output / "plan.json"
    p.write_text(json.dumps(plan, indent=2) + "\n")
    plan_hash = digest(p)
    p.with_suffix(".sha256").write_text(plan_hash + "\n")
    predictions, results = evaluate_bias(raw, plan["folds"], args.config)
    assert results["scored_trials"] == plan["scored_trials"]
    predictions.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(results["per_subject"]).to_csv(args.output / "subject_scores.csv", index=False)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    recovery = []
    for regime in plan["recovery"]["regimes"]:
        for seed in plan["recovery"]["seeds"]:
            simulated = simulate(regime, seed)
            folds = make_folds(simulated, ComparisonConfig(folds=3))
            _, summary = evaluate_bias(simulated, folds, args.config)
            recovery.append({"regime": regime, "seed": seed, **summary})
            print(f"recovery {regime} seed={seed} completed", flush=True)
    if digest(p) != plan_hash or any(digest(Path(path)) != value for path,value in {**inputs, **code}.items()):
        raise RuntimeError("Frozen experiment inputs or source changed")
    (args.output / "recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
    (args.output / "complete.json").write_text(json.dumps({"plan_sha256": plan_hash, "hash_checks": True,
                                                        "recovery_runs": len(recovery)}) + "\n")
    print(json.dumps(results["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    run(tyro.cli(Args))
