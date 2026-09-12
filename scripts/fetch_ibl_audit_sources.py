"""Cache public ALF trial arrays and identities for an explicit existing reference manifest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tyro

__all__ = ["Args", "fetch"]


@dataclass(slots=True)
class Args:
    """Explicit known-session inputs; never discover or download a confirmation cohort here."""

    manifest: Path = Path("data/ibl/reference.manifest.json")
    output: Path = Path("runs/ibl_source_audit")
    base_url: str = "https://openalyx.internationalbrainlab.org"


def fetch(args: Args) -> None:
    """Save raw source arrays and minimal identities, with resumable per-session downloads."""
    from one.api import ONE  # Optional acquisition dependency, outside the runtime environment.

    one = ONE(base_url=args.base_url, username=os.environ.get("ONE_USERNAME", "intbrainlab"),
              password=os.environ.get("ONE_PASSWORD", "international"),
              cache_dir=args.output / "one_cache", silent=True)
    raw_dir = args.output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries, errors = [], []
    for index, session in enumerate(manifest["sessions"]):
        eid = session["session_id"]
        arrays_path, metadata_path = raw_dir / f"{eid}.npz", raw_dir / f"{eid}.json"
        try:
            if not (arrays_path.exists() and metadata_path.exists()):
                remote = one.alyx.rest("sessions", "read", id=eid)
                identity = {key: remote[key] for key in ("id", "subject", "lab", "start_time", "task_protocol")}
                if identity["id"] != eid or not identity["subject"] or not identity["lab"]:
                    raise ValueError("Missing or inconsistent identity.")
                trials = one.load_object(eid, "trials")
                np.savez_compressed(arrays_path, **{key: np.asarray(value) for key, value in trials.items()})
                identity.update({"source": args.base_url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                 "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest()})
                metadata_path.write_text(json.dumps(identity, indent=2) + "\n")
            identity = json.loads(metadata_path.read_text())
            if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != identity["arrays_sha256"]:
                raise ValueError("Cached arrays hash mismatch.")
            entries.append(identity)
            print(f"[{index + 1}/{len(manifest['sessions'])}] cached {eid}", flush=True)
        except Exception as exc:
            errors.append({"session_id": eid, "error": str(exc)})
            print(f"[{index + 1}] FAILED {eid}: {type(exc).__name__}", flush=True)
        (args.output / "source_manifest.json").write_text(json.dumps({
            "reference_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source": args.base_url, "sessions": entries, "errors": errors,
        }, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"{len(errors)} sources unavailable; inspect source_manifest.json and resume.")


if __name__ == "__main__":
    fetch(tyro.cli(Args))
