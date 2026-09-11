"""Freeze models and rules, download only reserved subjects, then score once without tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from eval.history_audit import AuditConfig, MODELS, design_matrix, fit_logistic, prepare_trials
from eval.ibl_replication import ReplicationRules, convert_source, score_cohort
from eval.metrics import load_trials
from eval.schema_validator import validate_file

__all__ = ["Args", "run"]

CODE_FILES = [Path("scripts/run_ibl_replication.py"), Path("eval/ibl_replication.py"),
              Path("eval/history_audit.py"), Path("eval/metrics.py"), Path("eval/schema_validator.py"),
              Path("docs/REPLICATION_PROTOCOL.md")]


@dataclass(slots=True)
class Args:
    """Stages intentionally separate the analysis freeze from test-data access."""

    phase: Literal["freeze", "download", "score"] = "freeze"
    output: Path = Path("runs/ibl_replication_v1")
    reference: Path = Path("runs/ibl_source_reconciled/reference.ndjson")
    cohort: Path = Path("runs/ibl_replication_plan/confirmation_manifest.json")
    identities: Path = Path("runs/ibl_source_audit/source_manifest.json")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run(args: Args) -> None:
    """Enforce hash-bound stage ordering; downloads never fit or score a model."""
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_plan.json"
    if args.phase == "freeze":
        if plan_path.exists() or (args.output / "raw").exists():
            raise ValueError("Freeze already exists or test access began; use the existing plan.")
        cohort = json.loads(args.cohort.read_text())
        identities = json.loads(args.identities.read_text())
        known = {s["subject"] for s in identities["sessions"]} | set(cohort["excluded_subjects"])
        selected = cohort["sessions"]
        subjects = {s["subject"] for s in selected}
        if subjects & known or len(subjects) != len(selected):
            raise ValueError("Test subjects overlap development or have duplicate sessions.")
        rules, analysis = ReplicationRules(), AuditConfig()
        data = prepare_trials(load_trials(args.reference), analysis)
        models = {}
        for model in MODELS:
            x, names = design_matrix(data, model)
            beta = fit_logistic(x, data.choice_right.to_numpy(), analysis.l2)
            models[model] = {"features": names, "coefficients": list(map(float, beta))}
        models_path = args.output / "frozen_models.json"
        _write(models_path, models)
        plan = {"frozen_at": datetime.now(timezone.utc).isoformat(), "status": "internal_freeze_before_reserved_trial_download",
                "cohort": selected, "excluded_subjects": sorted(known), "analysis": asdict(analysis), "rules": asdict(rules),
                "training_reference_sha256": _hash(args.reference), "cohort_manifest_sha256": _hash(args.cohort),
                "identity_manifest_sha256": _hash(args.identities), "models_sha256": _hash(models_path),
                "code_sha256": {str(p): _hash(p) for p in CODE_FILES},
                "versions": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pandas", "pydantic")},
                "primary": "Mean across test subjects of outcome_history NLL minus evidence_history NLL; lower NLL is better.",
                "decision": "At least 40 eligible subjects, mean gain >= 0.0005 nats/trial, and subject bootstrap 95% lower bound > 0.",
                "threshold_rationale": "A fixed practical screen around half the exploratory gain, not an established biological threshold.",
                "sampling": "60 reserved subjects, one latest standard biased session each; no replacements after QC.",
                "exclusions": "Preset trial/easy-trial/accuracy gates; source-integrity errors exclude a session and are reported.",
                "no_tuning": "No hyperparameter selection, retraining, threshold changes, or replacement cohort based on test outcomes.",
                "limits": "Internal prospective freeze only; no external preregistration or committed-protocol claim."}
        _write(plan_path, plan)
        (args.output / "frozen_plan.sha256").write_text(_hash(plan_path) + "\n")
        print(f"Frozen {len(models)} models before access to {len(subjects)} test subjects.")
        return
    plan = json.loads(plan_path.read_text())
    if _hash(plan_path) != (args.output / "frozen_plan.sha256").read_text().strip():
        raise ValueError("Frozen protocol changed.")
    if any(_hash(Path(p)) != h for p, h in plan["code_sha256"].items()):
        raise ValueError("Analysis code changed after freeze.")
    if _hash(args.output / "frozen_models.json") != plan["models_sha256"]:
        raise ValueError("Frozen model coefficients changed.")
    raw_dir = args.output / "raw"
    if args.phase == "download":
        from one.api import ONE
        one = ONE(base_url="https://openalyx.internationalbrainlab.org",
                  username=os.environ.get("ONE_USERNAME", "intbrainlab"),
                  password=os.environ.get("ONE_PASSWORD", "international"),
                  cache_dir=args.output / "one_cache", silent=True)
        raw_dir.mkdir(exist_ok=True)
        acquired = []
        for i, session in enumerate(plan["cohort"]):
            eid = session["id"]
            path = raw_dir / f"{eid}.npz"
            if not path.exists():
                arrays = one.load_object(eid, "trials")
                np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
            acquired.append({"id": eid, "sha256": _hash(path)})
            _write(args.output / "acquisition.json", acquired)
            print(f"Downloaded {i+1}/{len(plan['cohort'])}", flush=True)
        return
    if (args.output / "metrics.json").exists():
        raise ValueError("Cohort already scored; inspect the saved result without retuning.")
    acquired = {s["id"]: s["sha256"] for s in json.loads((args.output / "acquisition.json").read_text())}
    if set(acquired) != {s["id"] for s in plan["cohort"]}:
        raise ValueError("Acquisition incomplete; resume without replacing subjects.")
    rules, analysis = ReplicationRules(**plan["rules"]), AuditConfig(**plan["analysis"])
    eligible_rows, checks = [], []
    subject_map = {}
    for session in plan["cohort"]:
        path = raw_dir / f"{session['id']}.npz"
        if _hash(path) != acquired[session["id"]]:
            raise ValueError("Downloaded source hash changed.")
        try:
            with np.load(path, allow_pickle=False) as arrays:
                rows, check = convert_source(dict(arrays), session["id"], rules)
        except ValueError as exc:
            check = {"session_id": session["id"], "eligible": False, "source_error": str(exc)}
        if check["eligible"]:
            eligible_rows.extend(rows)
            subject_map[session["id"]] = session["subject"]
        checks.append({"subject": session["subject"], "lab": session["lab"], **check})
    _write(args.output / "eligibility.json", checks)
    if not eligible_rows:
        _write(args.output / "metrics.json", {"status": "insufficient_eligible_subjects", "n_subjects": 0, "criterion_met": False})
        return
    trial_path = args.output / "trials.ndjson"
    trial_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in eligible_rows))
    validate_file(trial_path)
    models = json.loads((args.output / "frozen_models.json").read_text())
    metrics = score_cohort(load_trials(trial_path), models, subject_map, analysis, rules)
    metrics.update({"frozen_plan_sha256": _hash(plan_path), "trials_sha256": _hash(trial_path),
                    "acquisition_sha256": _hash(args.output / "acquisition.json")})
    _write(args.output / "metrics.json", metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_subject"}, indent=2))


if __name__ == "__main__":
    run(tyro.cli(Args))
