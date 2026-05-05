"""Shared helpers for experiment sweep scripts."""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def slug(value: object, *, negative: str = "m") -> str:
    """Format values for stable run-directory names."""
    return str(value).replace(".", "p").replace("-", negative)


def run_command(cmd: Sequence[str], *, dry_run: bool) -> int:
    """Run a command, returning the exit code."""
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write rows to a CSV file using the first row as the field contract."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarise_behavior_metrics(
    run_dir: Path,
    *,
    include_sticky: bool = False,
    include_chrono_overshoot: bool = False,
) -> dict[str, object]:
    """Extract common behavioral metrics from a completed run."""
    summary: dict[str, object] = {}
    metrics = load_json(run_dir / "metrics.json")
    if not metrics or "metrics" not in metrics:
        return summary

    data = metrics["metrics"]
    if not isinstance(data, dict):
        return summary

    psychometric = _as_mapping(data.get("psychometric"))
    chronometric = _as_mapping(data.get("chronometric"))
    history = _as_mapping(data.get("history"))
    basic = _as_mapping(data.get("basic"))
    quality = _as_mapping(data.get("quality"))

    summary.update(
        {
            "psych_slope": psychometric.get("slope"),
            "psych_bias": psychometric.get("bias"),
            "lapse_low": psychometric.get("lapse_low"),
            "lapse_high": psychometric.get("lapse_high"),
            "chrono_slope": chronometric.get("slope_ms_per_unit"),
            "chrono_r2": chronometric.get("r_squared"),
            "win_stay": history.get("win_stay"),
            "lose_shift": history.get("lose_shift"),
            "commit_rate": basic.get("commit_rate"),
            "chrono_ok": quality.get("chronometric_ok"),
            "degenerate": quality.get("degenerate"),
        }
    )
    if include_sticky:
        summary["sticky_choice"] = history.get("sticky_choice")
    if include_chrono_overshoot:
        summary["chrono_overshoot"] = quality.get("chrono_overshoot")
    return summary


def format_mean_std(values: Sequence[object], *, decimals: int = 2) -> str:
    """Format numeric values as mean +/- std for sweep summaries."""
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return "N/A"
    mean = statistics.mean(numeric)
    std = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


__all__ = [
    "format_mean_std",
    "load_json",
    "run_command",
    "slug",
    "summarise_behavior_metrics",
    "write_csv",
]
