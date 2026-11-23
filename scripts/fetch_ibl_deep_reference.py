"""Acquire every standard session of the existing development animals; never a new cohort."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Literal

import numpy as np
import tyro

__all__ = ["Args", "FAMILIES", "protocol_family", "select_sessions", "run"]

# Exact versioned standard protocols only. Optogenetic, RPE, habituation and
# other named variants never match, mirroring the frozen replication rule.
FAMILIES = {
    "biased": re.compile(r"_iblrig_tasks_biasedChoiceWorld\d+(?:\.\d+)*"),
    "training": re.compile(r"_iblrig_tasks_trainingChoiceWorld\d+(?:\.\d+)*"),
    "ephys": re.compile(r"_iblrig_tasks_ephysChoiceWorld\d+(?:\.\d+)*"),
}
IDENTITY_KEYS = ("id", "subject", "lab", "start_time", "task_protocol")


@dataclass(slots=True)
class Args:
    """Stages run separately: inventory and plan use metadata only; download loads trials."""

    stage: Literal["inventory", "plan", "download", "verify"]
    source_manifest: Path = Path("runs/ibl_source_audit/source_manifest.json")
    reserved_manifest: Path = Path("runs/ibl_replication_plan/confirmation_manifest.json")
    output: Path = Path("runs/ibl_deep_reference_v1")
    base_url: str = "https://openalyx.internationalbrainlab.org"
    shard_index: int = 0
    shard_count: int = 1


def protocol_family(task_protocol: str | None) -> str | None:
    """Return the standard protocol family, or None for any variant or missing protocol."""
    for family, pattern in FAMILIES.items():
        if task_protocol and pattern.fullmatch(task_protocol):
            return family
    return None


def select_sessions(
    inventory: list[dict[str, object]], development: set[str], reserved: set[str],
) -> list[dict[str, object]]:
    """Keep standard-protocol sessions of development animals; no outcome or QC filtering."""
    if development & reserved:
        raise ValueError("Development and reserved animals overlap.")
    selected: dict[str, dict[str, object]] = {}
    for session in inventory:
        subject = session["subject"]
        if subject in reserved:
            raise ValueError("Reserved animal present in the development inventory.")
        family = protocol_family(session.get("task_protocol"))
        if subject not in development or family is None or not session.get("lab"):
            continue
        entry = {key: session[key] for key in IDENTITY_KEYS} | {"family": family}
        if selected.setdefault(str(session["id"]), entry) != entry:
            raise ValueError("Conflicting duplicate session identity.")
    return sorted(selected.values(), key=lambda s: (s["subject"], s["start_time"], s["id"]))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _client(args: Args, cache: Path):
    from one.api import ONE  # Optional acquisition dependency, outside the runtime environment.

    return ONE(base_url=args.base_url, username=os.environ.get("ONE_USERNAME", "intbrainlab"),
               password=os.environ.get("ONE_PASSWORD", "international"), cache_dir=cache, silent=True)


def _cohorts(args: Args) -> tuple[set[str], set[str]]:
    source = json.loads(args.source_manifest.read_text())
    if source["errors"]:
        raise ValueError("Resolve source acquisition errors first.")
    development = {str(s["subject"]) for s in source["sessions"]}
    reserved = {str(s["subject"]) for s in json.loads(args.reserved_manifest.read_text())["sessions"]}
    return development, reserved


def _inventory(args: Args) -> None:
    """List session metadata for each development animal by name; request no trial arrays."""
    development, reserved = _cohorts(args)
    if development & reserved:
        raise ValueError("Development and reserved animals overlap.")
    path = args.output / "inventory.json"
    if path.exists():
        raise ValueError("Inventory exists; use a fresh output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    one = _client(args, args.output / "metadata_cache")
    rows = []
    for index, subject in enumerate(sorted(development)):
        for item in one.alyx.rest("sessions", "list", subject=subject, limit=500):
            if item["subject"] != subject:
                raise ValueError("Inventory returned a different animal.")
            rows.append({key: item.get(key) for key in IDENTITY_KEYS})
        print(f"[{index + 1}/{len(development)}] {len(rows)} sessions listed", flush=True)
    path.write_text(json.dumps({"source": args.base_url, "sessions": rows,
                                "retrieved_at": datetime.now(timezone.utc).isoformat()},
                               indent=1, sort_keys=True) + "\n")


def _plan(args: Args) -> None:
    """Freeze the session list and its inputs before any trial array is requested."""
    path = args.output / "plan.json"
    if path.exists():
        raise ValueError("Plan exists; it is never replaced.")
    development, reserved = _cohorts(args)
    inventory = json.loads((args.output / "inventory.json").read_text())["sessions"]
    sessions = select_sessions(inventory, development, reserved)
    variants = Counter(str(s.get("task_protocol")) for s in inventory if protocol_family(s.get("task_protocol")) is None)
    plan = {
        "status": "development animals only; exploratory; no confirmation cohort accessed",
        "source": args.base_url, "sessions": sessions,
        "counts": {"sessions": len(sessions), "subjects": len({s["subject"] for s in sessions}),
                   "labs": len({s["lab"] for s in sessions}),
                   "by_family": dict(sorted(Counter(str(s["family"]) for s in sessions).items())),
                   "development_subjects": len(development),
                   "development_subjects_without_standard_sessions":
                       sorted(development - {str(s["subject"]) for s in sessions})},
        "excluded_variant_protocol_sessions": dict(sorted(variants.items())),
        "selection": "Exact versioned biased/training/ephys ChoiceWorld; all sessions; no outcome or QC filtering.",
        "input_sha256": {str(p): _digest(p) for p in (args.source_manifest, args.reserved_manifest,
                                                      args.output / "inventory.json")},
        "script_sha256": _digest(Path(__file__)),
    }
    path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    path.with_suffix(".sha256").write_text(_digest(path) + "\n")
    print(json.dumps(plan["counts"], indent=2))


def _frozen_sessions(args: Args) -> list[dict[str, object]]:
    path = args.output / "plan.json"
    if _digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Plan changed after freezing.")
    return json.loads(path.read_text())["sessions"]


def _download(args: Args) -> None:
    """Save raw trial arrays per session with hashes; resumable; failures are recorded, not replaced."""
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard.")
    sessions = _frozen_sessions(args)[args.shard_index::args.shard_count]
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    one = _client(args, args.output / "one_cache" / f"shard_{args.shard_index}")
    for index, session in enumerate(sessions):
        eid = str(session["id"])
        arrays_path, metadata_path, error_path = (raw / f"{eid}{suffix}" for suffix in (".npz", ".json", ".error.json"))
        if arrays_path.exists() and metadata_path.exists():
            continue
        try:
            trials = one.load_object(eid, "trials")
            temporary = raw / f"{eid}.partial.npz"
            np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in trials.items()})
            temporary.replace(arrays_path)
            metadata_path.write_text(json.dumps(session | {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "arrays_sha256": _digest(arrays_path)}, indent=1, sort_keys=True) + "\n")
            error_path.unlink(missing_ok=True)
        except Exception as exc:  # Unavailable sources are reported in the manifest.
            error_path.write_text(json.dumps({"id": eid, "error": f"{type(exc).__name__}: {exc}"[:500]}) + "\n")
        if (index + 1) % 25 == 0:
            print(f"shard {args.shard_index}: {index + 1}/{len(sessions)}", flush=True)
    print(f"shard {args.shard_index}: finished {len(sessions)}", flush=True)


def _verify(args: Args) -> None:
    """Rehash every cached source and write one manifest covering all planned sessions."""
    raw, entries, errors = args.output / "raw", [], []
    for session in _frozen_sessions(args):
        eid = str(session["id"])
        arrays_path, metadata_path, error_path = (raw / f"{eid}{suffix}" for suffix in (".npz", ".json", ".error.json"))
        if arrays_path.exists() and metadata_path.exists():
            identity = json.loads(metadata_path.read_text())
            if _digest(arrays_path) != identity["arrays_sha256"]:
                raise ValueError(f"Cached arrays hash mismatch: {eid}")
            if any(identity[key] != session[key] for key in (*IDENTITY_KEYS, "family")):
                raise ValueError(f"Cached identity differs from plan: {eid}")
            entries.append(identity)
        else:
            reason = json.loads(error_path.read_text())["error"] if error_path.exists() else "not attempted"
            errors.append({"id": eid, "subject": session["subject"], "family": session["family"], "error": reason})
    manifest = {"plan_sha256": _digest(args.output / "plan.json"), "source": args.base_url,
                "sessions": entries, "errors": errors,
                "counts": {"cached": len(entries), "unavailable": len(errors),
                           "cached_by_family": dict(sorted(Counter(e["family"] for e in entries).items())),
                           "cached_subjects": len({e["subject"] for e in entries})}}
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


def run(args: Args) -> None:
    """Dispatch one explicit stage."""
    {"inventory": _inventory, "plan": _plan, "download": _download, "verify": _verify}[args.stage](args)


if __name__ == "__main__":
    run(tyro.cli(Args))
