"""Development-cohort sensitivity check with optogenetic and RPE variant sessions removed."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import tyro

from eval.history_audit import AuditConfig, audit_history
from eval.metrics import load_trials
from scripts.compare_generative_history import evaluate, load_development, save, summarize
from scripts.fetch_ibl_deep_reference import protocol_family

__all__ = ["Args", "split_sessions", "run"]


@dataclass(slots=True)
class Args:
    """`prepare` writes filtered inputs and the history audit; `generative` refits the latent comparison."""

    stage: Literal["prepare", "generative"]
    reference: Path = Path("runs/ibl_source_reconciled/reference.ndjson")
    identities: Path = Path("runs/ibl_source_audit/source_manifest.json")
    output: Path = Path("runs/variant_sensitivity_20261123")
    competing_plan: Path = Path("runs/competing_history_novariant_20261123/plan.json")
    maxiter: int = 1000


def split_sessions(sessions: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split sessions into exact standard biased protocol and everything else."""
    kept = [s for s in sessions if protocol_family(s.get("task_protocol")) == "biased"]
    return kept, [s for s in sessions if protocol_family(s.get("task_protocol")) != "biased"]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(args: Args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory.")
    source = json.loads(args.identities.read_text())
    kept, excluded = split_sessions(source["sessions"])
    kept_ids = {s["id"] for s in kept}
    args.output.mkdir(parents=True, exist_ok=True)
    reference, manifest = args.output / "reference.ndjson", args.output / "source_manifest.json"
    with args.reference.open() as lines:
        reference.write_text("".join(line for line in lines if json.loads(line)["session_id"] in kept_ids))
    manifest.write_text(json.dumps(source | {"sessions": kept}, indent=1, sort_keys=True) + "\n")
    analysis = audit_history(load_trials(reference), AuditConfig(),
                             session_subjects={s["id"]: s["subject"] for s in kept})
    kept_subjects = {s["subject"] for s in kept}
    payloads = {
        # Same keys the competing-history runner checks before it accepts a reference.
        "config.json": {"candidate_sha256": _digest(reference), "source_manifest_sha256": _digest(manifest),
                        "input_sha256": {str(p): _digest(p) for p in (args.reference, args.identities)},
                        "script_sha256": _digest(Path(__file__)),
                        "status": "exploratory sensitivity check; previously inspected development animals"},
        "exclusion.json": {"rule": "Keep exact versioned biasedChoiceWorld; drop every other protocol.",
                           "kept_sessions": len(kept), "kept_subjects": len(kept_subjects),
                           "excluded_sessions": excluded,
                           "subjects_removed": sorted({s["subject"] for s in excluded} - kept_subjects)},
        "history_audit.json": analysis,
    }
    for name, payload in payloads.items():
        (args.output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"kept_sessions": len(kept), "excluded_sessions": len(excluded),
                      "kept_subjects": len(kept_subjects), "pooled_retry": analysis["pooled_retry"],
                      "comparisons": analysis["comparisons"]}, indent=2))


def _generative(args: Args) -> None:
    """Refit the development comparison only; recovery simulations are not repeated."""
    path = args.output / "generative_development.json"
    if path.exists():
        raise ValueError("Generative result exists; it is never replaced.")
    raw, previous = load_development(args.competing_plan)
    predictions, fits = evaluate(raw, previous["subject_folds"], args.maxiter)
    predictions.to_csv(args.output / "generative_predictions.csv", index=False)
    summary = summarize(predictions)
    save(path, {"summary": summary, "fits": fits, "competing_plan_sha256": _digest(args.competing_plan)})
    print(json.dumps({"gains": summary["gains"], "equal_animal_nll": summary["equal_animal_nll"],
                      "animals": summary["animals"], "trials": summary["trials"]}, indent=2))


def run(args: Args) -> None:
    """Dispatch one explicit stage."""
    {"prepare": _prepare, "generative": _generative}[args.stage](args)


if __name__ == "__main__":
    run(tyro.cli(Args))
