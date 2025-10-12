from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from eval.metrics import load_and_compute


@dataclass(slots=True)
class EvaluateArgs:
    run: Path | None = None
    log: Path | None = None
    out: Path | None = None


def _resolve_log_path(run: Path | None, log: Path | None) -> Path:
    if log is not None:
        return log
    if run is not None:
        return run / "trials.ndjson"
    raise SystemExit("You must provide either --run (with trials.ndjson) or --log path")


def _resolve_out_path(run: Path | None, log_path: Path, out: Path | None) -> Path:
    if out is not None:
        return out
    if run is not None:
        return run / "metrics.json"
    return log_path.parent / "metrics.json"


def main(args: EvaluateArgs) -> None:
    log_path = _resolve_log_path(args.run, args.log)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    metrics = load_and_compute(log_path)
    if not metrics:
        raise SystemExit(f"No metrics produced for log: {log_path}")

    out_path = _resolve_out_path(args.run, log_path, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "log": str(log_path),
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main(tyro.cli(EvaluateArgs))
