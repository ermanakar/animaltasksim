#!/usr/bin/env python
"""
Grid search utility for the hybrid DDM+LSTM agent on IBL 2AFC.

Launches combinations of drift_scale and noise initialization
to find parameters that steepen the psychometric curve.
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import tyro



@dataclass(slots=True)
class SweepArgs:
    run_root: Path = Path("runs/hybrid_sweep_ibl_drift_choice")
    """Root directory where sweep runs are stored."""

    drift_scales: Sequence[float] = (10.0, 20.0, 30.0)
    choice_weights: Sequence[float] = (0.5, 1.0, 1.5)

    seed: int = 42
    dry_run: bool = False


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarise_run(run_dir: Path) -> dict:
    summary: dict[str, object] = {"run_dir": str(run_dir)}

    metrics = load_json(run_dir / "metrics.json")
    if metrics and "metrics" in metrics:
        data = metrics["metrics"]
        summary.update(
            {
                "psychometric_slope": data.get("psychometric", {}).get("slope"),
                "psychometric_bias": data.get("psychometric", {}).get("bias"),
                "chronometric_slope": data.get("chronometric", {}).get("slope_ms_per_unit"),
                "history_win_stay": data.get("history", {}).get("win_stay"),
                "history_lose_shift": data.get("history", {}).get("lose_shift"),
                "chronometric_ok": data.get("quality", {}).get("chronometric_ok"),
                "degenerate": data.get("quality", {}).get("degenerate"),
            }
        )

    training_stats = load_json(run_dir / "training_metrics.json")
    if training_stats:
        summary["final_reward"] = training_stats.get("reward")
    return summary


def main(args: SweepArgs) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.drift_scales, args.choice_weights))
    summaries: list[dict[str, object]] = []

    for drift_scale, choice_w in combos:
        run_name = (
            f"ibl_drift{drift_scale}_choice{choice_w}_seed{args.seed}"
            .replace(".", "p")
            .replace("-", "m")
        )
        run_dir = args.run_root / run_name

        cmd_train = [
            sys.executable,
            "scripts/train_hybrid_curriculum.py",
            "--task=ibl_2afc",
            f"--output-dir={run_dir}",
            f"--seed={args.seed}",
            f"--drift-scale={drift_scale}",
        ]
        
        # Add the curriculum configs tailored for steep psychometric slopes
        cmd_train.append("--no-use-default-curriculum")
        cmd_train.append("--no-allow-early-stopping")
        
        cmd_train.extend(
            [
                "--phase1-epochs=15",
                "--phase1-choice-weight=0.0",
                "--phase1-rt-weight=1.0",
                "--phase1-history-weight=0.0",
                "--phase1-drift-magnitude-weight=0.5",
                "--phase2-epochs=10",
                f"--phase2-choice-weight={choice_w * 0.5}",
                "--phase2-rt-weight=0.8",
                "--phase2-history-weight=0.1",
                "--phase2-drift-magnitude-weight=0.5",
                "--phase3-epochs=10",
                f"--phase3-choice-weight={choice_w}",
                "--phase3-rt-weight=0.5",
                "--phase3-history-weight=0.2",
                "--phase3-drift-magnitude-weight=0.5",
            ]
        )
        
        print(f"\\n=== Starting {run_name} ===")
        run_command(cmd_train, dry_run=args.dry_run)

        cmd_eval = [
            sys.executable,
            "scripts/evaluate_agent.py",
            "--run",
            str(run_dir),
        ]
        run_command(cmd_eval, dry_run=args.dry_run)

        if not args.dry_run:
            summary = {
                "run": run_name,
                "drift_scale": drift_scale,
                "choice_weight": choice_w,
            }
            summary.update(summarise_run(run_dir))
            summaries.append(summary)

    if summaries and not args.dry_run:
        csv_path = args.run_root / "sweep_summary.csv"
        fieldnames = sorted({key for row in summaries for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        print(f"[INFO] Summary saved to {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(SweepArgs))
