"""Reserve metadata-only confirmation subjects, excluding every local IBL reference animal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import tyro

__all__ = ["Args", "select_cohort", "prepare"]


@dataclass(slots=True)
class Args:
    """Deterministic metadata sampling; this command never loads trial datasets."""

    source_manifest: Path = Path("runs/ibl_source_audit/source_manifest.json")
    output: Path = Path("runs/ibl_replication_plan")
    subjects: int = 60
    seed: int = 20260906
    base_url: str = "https://openalyx.internationalbrainlab.org"


def select_cohort(
    sessions: list[dict[str, str]], excluded_subjects: set[str], count: int, seed: int,
) -> list[dict[str, str]]:
    """Select hash-ordered subjects and their latest standard biased session, without outcomes."""
    if count < 1:
        raise ValueError("At least one subject is required.")
    latest: dict[str, dict[str, str]] = {}
    for session in sessions:
        if not re.fullmatch(r"_iblrig_tasks_biasedChoiceWorld\d+(?:\.\d+)*", session["task_protocol"]):
            continue
        subject = session["subject"]
        if not subject or subject in excluded_subjects or not session["lab"]:
            continue
        previous = latest.get(subject)
        if previous is None or (session["start_time"], session["id"]) > (previous["start_time"], previous["id"]):
            latest[subject] = session
    ordered = sorted(latest, key=lambda subject: hashlib.sha256(f"{seed}:{subject}".encode()).hexdigest())
    return [latest[subject] for subject in ordered[:count]]


def prepare(args: Args) -> None:
    """Fetch identities and the metadata sampling frame; do not request behavioral arrays."""
    from one.api import ONE

    source = json.loads(args.source_manifest.read_text())
    if source["errors"]:
        raise ValueError("Resolve source acquisition errors first.")
    one = ONE(base_url=args.base_url, username=os.environ.get("ONE_USERNAME", "intbrainlab"),
              password=os.environ.get("ONE_PASSWORD", "international"),
              cache_dir=args.output / "metadata_cache", silent=True)
    identities = {s["id"]: s for s in source["sessions"]}
    # Include legacy datasets too: excluding only the adopted 120 would leak old animals.
    reference_paths = sorted(Path("data/ibl").glob("*.ndjson"))
    observed_ids = set()
    reference_hashes = {}
    for path in reference_paths:
        reference_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        observed_ids.update(json.loads(line)["session_id"] for line in path.open() if line.strip())
    for eid in sorted(observed_ids - set(identities)):
        remote = one.alyx.rest("sessions", "read", id=eid)
        identities[eid] = {key: remote[key] for key in ("id", "subject", "lab", "start_time", "task_protocol")}
    if any(not item["subject"] or not item["lab"] for item in identities.values()):
        raise ValueError("Unresolved historical subject identity.")
    frame = one.alyx.rest("sessions", "list", task_protocol="biasedChoiceWorld", limit=500)
    allowed_keys = ("id", "subject", "lab", "start_time", "task_protocol")
    sessions = [{key: item[key] for key in allowed_keys} for item in frame]
    excluded = {item["subject"] for item in identities.values()}
    selected = select_cohort(sessions, excluded, args.subjects, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    frame_text = json.dumps(sessions, sort_keys=True, indent=2) + "\n"
    (args.output / "sampling_frame.json").write_text(frame_text)
    manifest = {
        "status": "metadata_only_reserved_not_downloaded_or_scored",
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "requested_subjects": args.subjects, "selected_subjects": len(selected),
        "source": args.base_url, "sampling_frame_sessions": len(sessions),
        "sampling_frame_sha256": hashlib.sha256(frame_text.encode()).hexdigest(),
        "reference_hashes": reference_hashes, "excluded_subjects": sorted(excluded),
        "historical_identities": [{key: item[key] for key in allowed_keys} for item in identities.values()],
        "sessions": selected,
        "selection": "Hash-order subjects; latest exact standard biasedChoiceWorld session per subject; no outcome/QC filtering.",
        "limits": ["Unseen relative to the inventoried local references, not a claim that no earlier fetch ever inspected them.",
                   "This is an internal prospective plan, not externally preregistered research.",
                   "No replacement of failed QC subjects based on behavioral results; assess prespecified cohort sufficiency."],
    }
    (args.output / "confirmation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "sampling_frame_sessions", "selected_subjects")}, indent=2))
    print(f"Excluded {len(excluded)} previously represented animals across {len(observed_ids)} reference sessions.")


if __name__ == "__main__":
    prepare(tyro.cli(Args))
