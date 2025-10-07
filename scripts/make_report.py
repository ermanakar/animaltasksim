from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from eval.metrics import load_and_compute
from eval.report import build_report


@dataclass(slots=True)
class ReportArgs:
    run: Path
    log: Path | None = None
    metrics: Path | None = None
    out: Path | None = None
    title: str = "AnimalTaskSim Report"


def main(args: ReportArgs) -> None:
    log_path = args.log or args.run / "trials.ndjson"
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    metrics_data = None
    if args.metrics and args.metrics.exists():
        payload = json.loads(args.metrics.read_text(encoding="utf-8"))
        metrics_data = payload.get("metrics", payload)
    else:
        metrics_data = load_and_compute(log_path)

    out_path = args.out or args.run / "report.html"
    build_report(log_path, out_path, title=args.title, metrics=metrics_data)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main(tyro.cli(ReportArgs))
