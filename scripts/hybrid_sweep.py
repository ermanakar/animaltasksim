#!/usr/bin/env python
"""
Grid search utility for the hybrid DDM+LSTM agent on macaque RDM.

Launches combinations of hyperparameters, evaluates runs, and
collects summary metrics into a CSV file.
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import tyro

from animaltasksim.config import ProjectPaths


@dataclass(slots=True)
class SweepArgs:
    run_root: Path = Path("runs/hybrid_sweep_rdm")
    """Root directory where sweep runs are stored."""

    wfpt_warmup_epochs: Sequence[int] = (5, 10)
    choice_loss_weights: Sequence[float] = (0.5, 1.0)
    history_loss_weights: Sequence[float] = (0.0, 0.1)
    rt_penalty_weights: Sequence[float] = (0.5,)

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
                "chronometric_intercept": data.get("chronometric", {}).get("intercept_ms"),
                "history_win_stay": data.get("history", {}).get("win_stay"),
                "history_lose_shift": data.get("history", {}).get("lose_shift"),
                "history_sticky": data.get("history", {}).get("sticky_choice"),
            }
        )

    training_stats = load_json(run_dir / "training_metrics.json")
    if training_stats:
        summary["final_reward"] = training_stats.get("reward")
    return summary


def main(args: SweepArgs) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)

    combos = list(
        itertools.product(
            args.wfpt_warmup_epochs,
            args.choice_loss_weights,
            args.history_loss_weights,
            args.rt_penalty_weights,
        )
    )

    summaries: list[dict[str, object]] = []
    reference_log = ProjectPaths.from_cwd().data / "macaque" / "reference.ndjson"

    for wfpt_epochs, choice_w, history_w, rt_w in combos:
        run_name = (
            f"rdm_wfpt{wfpt_epochs}_choice{choice_w}_history{history_w}_rt{rt_w}_seed{args.seed}"
            .replace(".", "p")
            .replace("-", "m")
        )
        run_dir = args.run_root / run_name

        cmd_train = [
            "python",
            "scripts/train_hybrid_curriculum.py",
            f"--reference-log={reference_log}",
            f"--output-dir={run_dir}",
            f"--seed={args.seed}",
        ]

        cmd_train.append("--no-use-default-curriculum")
        cmd_train.extend(
            [
                f"--phase1-epochs={wfpt_epochs}",
                "--phase1-choice-weight=0.0",
                "--phase1-rt-weight=1.0",
                "--phase1-history-weight=0.0",
                "--phase1-drift-supervision-weight=0.5",
                f"--phase2-epochs={max(5, wfpt_epochs // 2)}",
                f"--phase2-choice-weight={choice_w * 0.3}",
                "--phase2-rt-weight=0.8",
                f"--phase2-history-weight={history_w * 0.5}",
                "--phase2-drift-supervision-weight=0.3",
                "--phase3-epochs=10",
                f"--phase3-choice-weight={choice_w}",
                f"--phase3-rt-weight={rt_w if rt_w > 0 else 0.5}",
                f"--phase3-history-weight={history_w}",
                "--phase3-drift-supervision-weight=0.1",
            ]
        )
        run_command(cmd_train, dry_run=args.dry_run)

        cmd_eval = [
            "python",
            "scripts/evaluate_agent.py",
            "--run",
            str(run_dir),
        ]
        run_command(cmd_eval, dry_run=args.dry_run)

        cmd_dash = [
            "python",
            "scripts/make_dashboard.py",
            "--opts.agent-log",
            str(run_dir / "trials.ndjson"),
            "--opts.reference-log",
            str(reference_log),
            "--opts.output",
            str(run_dir / "dashboard.html"),
            "--opts.agent-name",
            run_name,
            "--opts.reference-name",
            "Roitman & Shadlen",
        ]
        run_command(cmd_dash, dry_run=args.dry_run)

        if not args.dry_run:
            summary = {
                "run": run_name,
                "phase1_epochs": wfpt_epochs,
                "phase3_choice_weight": choice_w,
                "phase3_history_weight": history_w,
                "phase3_rt_weight": rt_w,
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
