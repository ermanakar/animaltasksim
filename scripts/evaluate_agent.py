from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from eval.metrics import load_and_compute


@dataclass(slots=True)
class EvaluateArgs:
    run: Path
    log: Path | None = None
    out: Path | None = None


def main(args: EvaluateArgs) -> None:
    log_path = args.log or args.run / "trials.ndjson"
    metrics = load_and_compute(log_path)
    if not metrics:
        raise SystemExit(f"No metrics produced for log: {log_path}")

    out_path = args.out or args.run / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "log": str(log_path),
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(tyro.cli(EvaluateArgs))
