"""Verify the adopted reference against cached public sources and evaluate a separate candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import tyro

from eval.history_audit import AuditConfig, audit_history
from eval.ibl_source_audit import reconcile_session
from eval.metrics import load_trials
from eval.schema_validator import validate_file

__all__ = ["Args", "run"]


@dataclass(slots=True)
class Args:
    """Retain the adopted file; all candidate artifacts go to a separate directory."""

    reference: Path = Path("data/ibl/reference.ndjson")
    manifest: Path = Path("data/ibl/reference.manifest.json")
    sources: Path = Path("runs/ibl_source_audit")
    output: Path = Path("runs/ibl_source_reconciled")
    analysis: AuditConfig = field(default_factory=AuditConfig)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: Args) -> dict[str, object]:
    """Fail closed on unexplained mismatches; emit corrections and subject-level diagnostics."""
    source_manifest_path = args.sources / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest["errors"] or source_manifest["reference_manifest_sha256"] != _hash(args.manifest):
        raise ValueError("Incomplete or mismatched source acquisition.")
    source_sessions = {s["id"]: s for s in source_manifest["sessions"]}
    manifest = json.loads(args.manifest.read_text())
    expected_ids = {s["session_id"] for s in manifest["sessions"]}
    if expected_ids != set(source_sessions):
        raise ValueError("Source session set differs from the adopted manifest.")
    rows: dict[str, list[dict]] = {eid: [] for eid in expected_ids}
    for line in args.reference.open():
        row = json.loads(line)
        rows[row["session_id"]].append(row)
    all_candidate, checks = [], []
    for eid in sorted(rows):
        raw_path = args.sources / "raw" / f"{eid}.npz"
        if _hash(raw_path) != source_sessions[eid]["arrays_sha256"]:
            raise ValueError(f"Source hash mismatch for {eid}.")
        with np.load(raw_path, allow_pickle=False) as arrays:
            candidate, check = reconcile_session(rows[eid], dict(arrays))
        all_candidate.extend(candidate)
        checks.append({"session_id": eid, "subject": source_sessions[eid]["subject"],
                       "lab": source_sessions[eid]["lab"], **check})
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output / "reference.ndjson"
    if candidate_path.resolve() == args.reference.resolve():
        raise ValueError("Cannot overwrite the adopted reference.")
    candidate_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in all_candidate))
    validate_file(candidate_path)
    session_subjects = {eid: s["subject"] for eid, s in source_sessions.items()}
    analysis = audit_history(load_trials(candidate_path), args.analysis, session_subjects=session_subjects)
    report = {
        "status": "source_reconciled_candidate_not_adopted", "n_sessions": len(rows),
        "n_subjects": len(set(session_subjects.values())),
        "n_labs": len({s["lab"] for s in source_sessions.values()}),
        "n_trials": len(all_candidate), "raw_omissions": sum(s["raw_omissions"] for s in checks),
        "misclassified_omissions": sum(s["misclassified_omissions"] for s in checks),
        "sessions_affected": sum(s["misclassified_omissions"] > 0 for s in checks), "sessions": checks,
    }
    config = {"reference": str(args.reference), "reference_sha256": _hash(args.reference),
              "candidate_sha256": _hash(candidate_path), "source_manifest_sha256": _hash(source_manifest_path),
              "analysis": asdict(args.analysis),
              "code_sha256": {str(p): _hash(p) for p in [Path(__file__), Path("eval/ibl_source_audit.py"),
                                                       Path("eval/history_audit.py"), Path("eval/metrics.py")]}}
    for name, payload in (("reconciliation.json", report), ("metrics.json", analysis), ("config.json", config)):
        (args.output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sessions"}, indent=2))
    print(json.dumps(analysis["comparisons"], indent=2))
    return report


if __name__ == "__main__":
    run(tyro.cli(Args))
