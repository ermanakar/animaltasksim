"""Freeze and run competing-history comparisons on the corrected development cohort."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import pandas as pd
import tyro

from eval.competing_history import ComparisonConfig, compare, make_folds, prepare_comparison
from eval.metrics import load_trials

__all__ = ["Args", "run"]


@dataclass(slots=True)
class Args:
    """New output only; previously scored replication animals stay excluded."""

    reference: Path = Path("runs/ibl_source_reconciled/reference.ndjson")
    identities: Path = Path("runs/ibl_source_audit/source_manifest.json")
    reconciliation: Path = Path("runs/ibl_source_reconciled/config.json")
    reserved: Path = Path("docs/results/replication_freeze_2026-09-06.json")
    output: Path = Path("runs/competing_history_20260912")
    analysis: ComparisonConfig = field(default_factory=ComparisonConfig)


def digest(path: Path) -> str:
    """Hash frozen bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: Args) -> None:
    """Freeze folds, dependencies, and contrasts before evaluating the fixed models."""
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    reconciliation = json.loads(args.reconciliation.read_text())
    if digest(args.reference) != reconciliation["candidate_sha256"]:
        raise RuntimeError("Corrected development reference hash mismatch")
    if digest(args.identities) != reconciliation["source_manifest_sha256"]:
        raise RuntimeError("Identity manifest hash mismatch")
    source = json.loads(args.identities.read_text())
    identity = {r["id"]: r for r in source["sessions"]}
    if source["errors"] or len(identity) != len(source["sessions"]):
        raise ValueError("Incomplete or duplicate identities")
    reserved = json.loads(args.reserved.read_text())
    # Snapshot structure is checked explicitly rather than silently assuming no overlap.
    reserved_subjects = {r["subject"] for r in reserved["plan"]["cohort"]}
    raw = load_trials(args.reference)
    if not set(raw.session_id) <= set(identity):
        raise ValueError("Missing source identity")
    raw["subject"] = raw.session_id.map(lambda s: identity[s]["subject"])
    raw["lab"] = raw.session_id.map(lambda s: identity[s]["lab"])
    if raw[["subject", "lab"]].isna().any().any() or raw.subject.isin(reserved_subjects).any():
        raise ValueError("Missing identity or reserved cohort overlap")
    if (raw.groupby("subject").lab.nunique() != 1).any():
        raise ValueError("Subject occurs in multiple labs")
    data = prepare_comparison(raw, args.analysis)
    folds = make_folds(data, args.analysis)
    lab_folds = make_folds(data, args.analysis, "lab")
    inputs = {str(p): digest(p) for p in [args.reference, args.identities, args.reconciliation, args.reserved]}
    code = {str(p): digest(p) for p in [Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
            Path("eval/competing_history.py"), Path("eval/history_audit.py"),
            Path("eval/metrics.py"), Path("eval/schema_validator.py")]}
    plan = {"settings": {**asdict(args), **{k: str(getattr(args, k)) for k in
            ("reference", "identities", "reconciliation", "reserved", "output")}},
            "status": "exploratory_previously_inspected_development_cohort", "input_sha256": inputs,
            "source_sha256": code, "subject_folds": folds, "leave_one_lab_out": lab_folds,
            "subjects": sorted(data.subject.unique()), "reserved_subjects_excluded": sorted(reserved_subjects),
            "input_trials": len(raw), "common_trials": len(data), "common_sessions": data.session_id.nunique(),
            "common_subjects": data.subject.nunique(), "common_labs": data.lab.nunique(),
            "sample": "Current and preceding five trials must be contiguous and committed; identical rows for every model.",
            "primary": "Equal-animal NLL(combined_history) minus NLL(combined_evidence), five animal-disjoint folds.",
            "secondary": "Evidence gain with one-step, five-lag, or learning-only controls; combined history gain; leave-one-lab-out refits for sensitivity. No selection of best model or rate.",
            "learner": "Two fixed-rate leaky estimates of inferred correct side at rates .05,.2. Current outcome updates only next prediction. Reset at session/gap/omission; hold unchanged on zero contrast. Not a Bayesian observer.",
            "fitting": "Trial-weighted summed logistic loss plus ridge1; equal-animal evaluation. Rates/lags/ridge fixed without validation tuning.",
            "limits": ["True block is an analyst control shared by all models; this is not an autonomous-agent benchmark.",
                       "A weak learning-state score cannot rule out animal block inference, since block is already controlled.",
                       "Overlapping history features and differently scaled ridge penalties do not identify unique mechanisms or match capacities.",
                       "Bootstrap intervals condition on overlapping fitted folds; omit training-population and lab-sampling uncertainty.",
                       "Multiple exploratory contrasts, previously examined cohort; no new confirmation or novelty claim.",
                       "Common five-trial eligibility changes the population relative to the previous one-step audit; compare only within this study.",
                       "Leaky learner is one deliberately simple alternative; failure does not rule out richer learning models."]}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    plan_digest = digest(plan_path)
    (args.output / "plan.sha256").write_text(plan_digest + "\n")
    for name, unit, selected in [("subject_folds", "subject", folds), ("lab_folds", "lab", lab_folds)]:
        predictions, result = compare(data, args.analysis, selected, unit)
        predictions.to_csv(args.output / f"{name}_predictions.csv", index=False)
        pd.DataFrame(result["per_subject"]).to_csv(args.output / f"{name}_subject_scores.csv", index=False)
        result.update({"plan_sha256": plan_digest, "limits": plan["limits"]})
        if digest(plan_path) != plan_digest or any(digest(Path(p)) != h for p, h in {**inputs, **code}.items()):
            raise RuntimeError("Frozen plan, input or code changed")
        (args.output / f"{name}_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"scheme": name, "contrasts": result["contrasts"]}, indent=2), flush=True)


if __name__ == "__main__":
    run(tyro.cli(Args))
