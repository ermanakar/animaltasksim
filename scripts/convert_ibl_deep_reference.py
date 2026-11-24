"""Convert the deep development reference to schema trials and check it against the reconciled reference."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from eval.ibl_deep_convert import ConversionError, compare_sessions, convert_session
from scripts.fetch_ibl_deep_reference import protocol_family

__all__ = ["Args", "run"]

CODE = (Path("scripts/convert_ibl_deep_reference.py"), Path("eval/ibl_deep_convert.py"), Path("eval/schema_validator.py"))


@dataclass(slots=True)
class Args:
    """`convert` writes per-animal trial files and a manifest; `gate` compares shared sessions once."""

    stage: Literal["convert", "gate"]
    source: Path = Path("runs/ibl_deep_reference_v1")
    output: Path = Path("runs/ibl_deep_converted_v1")
    reference: Path = Path("runs/ibl_source_reconciled/reference.ndjson")
    reference_identities: Path = Path("runs/ibl_source_audit/source_manifest.json")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert(args: Args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    source_path = args.source / "source_manifest.json"
    source = json.loads(source_path.read_text())
    ordered = sorted(source["sessions"], key=lambda s: (s["subject"], s["start_time"], s["id"]))
    number: Counter[str] = Counter()
    lines: dict[tuple[str, str], list[str]] = defaultdict(list)
    sessions = []
    for session in ordered:
        raw = args.source / "raw" / f"{session['id']}.npz"
        if _digest(raw) != session["arrays_sha256"]:
            raise RuntimeError(f"Source hash mismatch for {session['id']}")
        entry = {key: session[key] for key in ("id", "subject", "lab", "family", "task_protocol", "start_time", "arrays_sha256")}
        entry["session_number"] = number[session["subject"]]
        number[session["subject"]] += 1
        with np.load(raw, allow_pickle=False) as arrays:
            try:
                rows, summary = convert_session(dict(arrays), session["id"], session["family"])
            except ConversionError as reason:
                sessions.append(entry | {"status": "not_converted", "reason": str(reason)})
                continue
        lines[(session["family"], session["subject"])] += [json.dumps(row, separators=(",", ":")) + "\n" for row in rows]
        sessions.append(entry | {"status": "converted"} | summary)
    files = {}
    for (family, subject), content in sorted(lines.items()):
        path = args.output / family / f"{subject}.ndjson"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(content))
        files[f"{family}/{subject}.ndjson"] = {"sha256": _digest(path), "trials": len(content)}
    converted = [s for s in sessions if s["status"] == "converted"]
    counts = {"planned_cached": len(sessions), "converted": len(converted),
              "not_converted": dict(Counter(s["reason"] for s in sessions if s["status"] != "converted")),
              "converted_by_family": dict(sorted(Counter(s["family"] for s in converted).items())),
              "trials_by_family": {family: sum(s["trials"] for s in converted if s["family"] == family)
                                   for family in sorted({s["family"] for s in converted})},
              "subjects": len({s["subject"] for s in converted}),
              "meets_frozen_gate": sum(s["meets_frozen_gate"] for s in converted)}
    manifest = {"status": "structural conversion only; no quality filtering", "counts": counts,
                "source_manifest_sha256": _digest(source_path), "code_sha256": {str(p): _digest(p) for p in CODE},
                "files": files, "sessions": sessions}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n")
    print(json.dumps(counts, indent=2))


def _gate(args: Args) -> None:
    """Every shared standard biased session must match the reconciled reference trial for trial."""
    path = args.output / "gate.json"
    if path.exists():
        raise RuntimeError("Gate result exists; it is never replaced")
    manifest = json.loads((args.output / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        if _digest(args.output / name) != record["sha256"]:
            raise RuntimeError(f"Converted file changed: {name}")
    identities = json.loads(args.reference_identities.read_text())["sessions"]
    shared = sorted(s["id"] for s in identities if protocol_family(s.get("task_protocol")) == "biased")
    reference: dict[str, list[dict]] = defaultdict(list)
    for line in args.reference.open():
        row = json.loads(line)
        reference[row["session_id"]].append(row)
    converted = {s["id"]: s for s in manifest["sessions"] if s["status"] == "converted"}
    cache: dict[str, dict[str, list[dict]]] = {}
    results, trials = [], 0
    for eid in shared:
        if eid not in converted:
            results.append({"session_id": eid, "matched": False, "differences": [{"field": "session", "converted": None}]})
            continue
        name = f"{converted[eid]['family']}/{converted[eid]['subject']}.ndjson"
        if name not in cache:
            cache[name] = defaultdict(list)
            for line in (args.output / name).open():
                row = json.loads(line)
                cache[name][row["session_id"]].append(row)
        differences = compare_sessions(cache[name][eid], reference[eid])
        trials += len(reference[eid])
        results.append({"session_id": eid, "matched": not differences, "trials": len(reference[eid]),
                        "differences": differences[:20], "difference_count": len(differences)})
    failed = [r for r in results if not r["matched"]]
    report = {"passed": not failed and len(shared) > 0, "shared_sessions": len(shared), "matched_sessions": len(shared) - len(failed),
              "trials_compared": trials, "reference_sha256": _digest(args.reference),
              "manifest_sha256": _digest(args.output / "manifest.json"), "failed": failed}
    path.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failed"} | {"failed_sessions": len(failed)}, indent=2))


def run(args: Args) -> None:
    """Dispatch one explicit stage."""
    {"convert": _convert, "gate": _gate}[args.stage](args)


if __name__ == "__main__":
    run(tyro.cli(Args))
