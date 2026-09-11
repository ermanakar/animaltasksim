"""Run the fixed exploratory IBL history comparison without changing historical artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
from pathlib import Path

import tyro

from eval.history_audit import AuditConfig, audit_history
from eval.metrics import load_trials

__all__ = ["Args", "run"]


@dataclass(slots=True)
class Args:
    """Paths and settings for a separately versioned analysis artifact."""

    reference: Path = Path("data/ibl/reference.ndjson")
    manifest: Path = Path("data/ibl/reference.manifest.json")
    output: Path = Path("runs/ibl_history_refresh")
    analysis: AuditConfig = field(default_factory=AuditConfig)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: Args) -> dict[str, object]:
    """Validate source provenance and write settings, fold manifest, and analysis results."""
    manifest = json.loads(args.manifest.read_text())
    df = load_trials(args.reference)
    sessions = manifest["sessions"]
    if set(df.session_id) != {item["session_id"] for item in sessions}:
        raise ValueError("Reference and manifest session IDs differ.")
    counts = df.groupby("session_id").size().to_dict()
    if any(counts[item["session_id"]] != item["kept"] for item in sessions):
        raise ValueError("Reference and manifest trial counts differ.")
    if any(item["dropped_off_protocol_contrast"] or item["dropped_unclassified"] for item in sessions):
        raise ValueError("Importer dropped rows: recover original trial adjacency before history analysis.")
    config = {
        "reference": str(args.reference), "reference_sha256": _sha256(args.reference),
        "manifest": str(args.manifest), "manifest_sha256": _sha256(args.manifest),
        "analysis": asdict(args.analysis),
        "source_sha256": {str(path): _sha256(path) for path in
                          (Path(__file__), Path("eval/history_audit.py"), Path("eval/metrics.py"))},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "pydantic")},
    }
    result = audit_history(df, args.analysis)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("config.json", config), ("metrics.json", result)):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in
                     ("status", "eligible_trials", "pooled_retry", "qualified_session_retry", "comparisons")}, indent=2))
    print(f"Saved provenance and full results to {args.output}")
    return result


if __name__ == "__main__":
    run(tyro.cli(Args))
