#!/usr/bin/env python
"""
Grid search utility for the R-DDM agent.

Runs combinations of hyperparameters for either IBL or RDM tasks,
captures training/evaluation outputs, and writes a summary CSV.
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import tyro

from animaltasksim.config import ProjectPaths


@dataclass(slots=True)
class SweepArgs:
    """CLI arguments controlling the sweep."""

    task: str = "ibl_2afc"
    """Target task: 'ibl_2afc' or 'rdm_macaque'."""

    run_root: Path = field(default_factory=lambda: Path("runs/rddm_sweep"))
    """Directory to store all sweep runs."""

    drift_supervision_weights: Sequence[float] = (0.2, 0.6, 1.0)
    history_loss_weights: Sequence[float] = (0.1, 0.2, 0.3)
    history_ramp_epochs: Sequence[int] = (8, 16)
    prior_feature_scales: Sequence[float] = (0.0, 0.1)

    max_sessions: int = 10
    epochs: int = 40
    seed: int = 42
    rollout_trials: int = 1200

    choice_warmup_weight: float = 3.0
    warmup_epochs: int = 6
    enable_wfpt_epoch: int = 8
    enable_history_epoch: int = 12

    choice_kl_weight: float = 0.0
    choice_kl_target_slope: float = 6.0
    entropy_loss_weight: float = 0.0

    dry_run: bool = False
    """When true, only prints the planned commands without executing them."""


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    if dry_run:
        return
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarise_run(run_dir: Path, use_best: bool = True) -> dict:
    training_stats = load_json(run_dir / "training_best.json")
    metrics = load_json(run_dir / "metrics.json")
    row: dict[str, object] = {"run_dir": str(run_dir)}
    if training_stats:
        row.update(
            {
                "best_epoch": training_stats.get("epoch"),
                "best_accuracy": training_stats.get("accuracy"),
                "best_choice_loss": training_stats.get("choice_loss"),
            }
        )
    if metrics and "metrics" in metrics:
        m = metrics["metrics"]
        psycho = m.get("psychometric", {})
        chrono = m.get("chronometric", {})
        history = m.get("history", {})
        row.update(
            {
                "psychometric_slope": psycho.get("slope"),
                "psychometric_bias": psycho.get("bias"),
                "chronometric_slope": chrono.get("slope_ms_per_unit"),
                "chronometric_intercept": chrono.get("intercept_ms"),
                "history_win_stay": history.get("win_stay"),
                "history_lose_shift": history.get("lose_shift"),
                "history_sticky": history.get("sticky_choice"),
            }
        )
    row["used_best_log"] = use_best
    return row


def main(args: SweepArgs) -> None:
    combos = list(
        itertools.product(
            args.drift_supervision_weights,
            args.history_loss_weights,
            args.history_ramp_epochs,
            args.prior_feature_scales,
        )
    )
    if not combos:
        raise SystemExit("No hyperparameter combinations provided.")

    args.run_root.mkdir(parents=True, exist_ok=True)
    summaries: List[dict[str, object]] = []

    default_ref = (
        ProjectPaths.from_cwd().data / "macaque" / "reference.ndjson"
        if args.task == "rdm_macaque"
        else ProjectPaths.from_cwd().data / "ibl" / "reference.ndjson"
    )

    for idx, (drift_w, history_w, ramp, prior_scale) in enumerate(combos, start=1):
        run_name = (
            f"{args.task}_dw{drift_w:g}_hw{history_w:g}_hr{ramp}_ps{prior_scale:g}_seed{args.seed}"
            .replace(".", "p")
            .replace("-", "m")
        )
        run_dir = args.run_root / run_name
        if run_dir.exists() and not args.dry_run:
            print(f"[SKIP] {run_dir} already exists.")
            continue

        cmd_train = [
            "python",
            "scripts/train_r_ddm.py",
            "--task",
            args.task,
            "--run-dir",
            str(run_dir),
            "--reference-log",
            str(default_ref),
            "--epochs",
            str(args.epochs),
            "--max-sessions",
            str(args.max_sessions),
            "--seed",
            str(args.seed),
            "--rollout-trials",
            str(args.rollout_trials),
            "--choice-warmup-weight",
            str(args.choice_warmup_weight),
            "--warmup-epochs",
            str(args.warmup_epochs),
            "--enable-wfpt-epoch",
            str(args.enable_wfpt_epoch),
            "--enable-history-epoch",
            str(args.enable_history_epoch),
            "--history-ramp-epochs",
            str(ramp),
            "--history-loss-weight",
            str(history_w),
            "--drift-supervision-weight",
            str(drift_w),
            "--prior-feature-scale",
            str(prior_scale),
            "--choice-kl-weight",
            str(args.choice_kl_weight),
            "--choice-kl-target-slope",
            str(args.choice_kl_target_slope),
            "--entropy-loss-weight",
            str(args.entropy_loss_weight),
        ]
        run_command(cmd_train, dry_run=args.dry_run)

        cmd_eval = [
            "python",
            "scripts/evaluate_agent.py",
            "--run",
            str(run_dir),
            "--use-best",
        ]
        run_command(cmd_eval, dry_run=args.dry_run)

        cmd_dashboard = [
            "python",
            "scripts/make_dashboard.py",
            "--opts.agent-log",
            str(run_dir / "trials_best.ndjson"),
            "--opts.reference-log",
            str(default_ref),
            "--opts.output",
            str(run_dir / "dashboard.html"),
            "--opts.agent-name",
            f"R-DDM sweep {idx}",
            "--opts.reference-name",
            "Reference",
        ]
        run_command(cmd_dashboard, dry_run=args.dry_run)

        summaries.append(
            {
                "run": run_name,
                "task": args.task,
                "drift_supervision_weight": drift_w,
                "history_loss_weight": history_w,
                "history_ramp_epochs": ramp,
                "prior_feature_scale": prior_scale,
            }
        )

        run_summary = summarise_run(run_dir)
        summaries[-1].update(run_summary)

    if summaries and not args.dry_run:
        csv_path = args.run_root / "sweep_summary.csv"
        fieldnames = sorted({key for row in summaries for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        print(f"[INFO] Wrote summary to {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(SweepArgs))
