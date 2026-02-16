#!/usr/bin/env python
"""
Grid search utility for fine-tuning the R-DDM agent's chronometric performance.
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import tyro

from tqdm import tqdm

from animaltasksim.config import ProjectPaths


@dataclass(slots=True)
class SweepArgs:
    """CLI arguments controlling the sweep."""

    task: str = "rdm_macaque"
    """Target task: 'rdm_macaque'."""

    run_root: Path = field(default_factory=lambda: Path("runs/finetune_sweep"))
    """Directory to store all sweep runs."""

    non_decision_targets: Sequence[float] = (0.2, 0.3, 0.4)
    non_decision_reg_weights: Sequence[float] = (0.1, 0.2, 0.3)
    wfpt_loss_weights: Sequence[float] = (0.5, 1.0, 1.5)
    learning_rates: Sequence[float] = (1e-4, 5e-5, 1e-5)

    max_sessions: int = 10
    epochs: int = 40
    seed: int = 42
    rollout_trials: int = 1200

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
                "p_right_overall": metrics.get("p_right_overall"),
                "rt_variance": metrics.get("rt_variance"),
            }
        )
    row["used_best_log"] = use_best
    return row


def is_bad(metrics):
    q = metrics.get("quality", {})
    h = metrics.get("history", {})
    p_right = metrics.get("p_right_overall", None)
    rt_var = metrics.get("rt_variance", None)
    chrono_ok = q.get("chronometric_ok", False)

    return (
        q.get("degenerate", False)
        or (p_right is not None and (p_right < 0.05 or p_right > 0.95))
        or (h.get("win_stay", 0) > 0.95 and h.get("lose_shift", 1) < 0.05)
        or (rt_var is not None and rt_var < 2500)              # 50ms^2
        or not chrono_ok
    )

def main(args: SweepArgs) -> None:
    combos = list(
        itertools.product(
            args.non_decision_targets,
            args.non_decision_reg_weights,
            args.wfpt_loss_weights,
            args.learning_rates,
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

    for idx, (ndt, ndt_w, wfpt_w, lr) in enumerate(tqdm(combos, desc="Fine-tuning sweep"), start=1):
        run_name = (
            f"{args.task}_ndt{ndt:g}_ndtw{ndt_w:g}_wfptw{wfpt_w:g}_lr{lr:g}_seed{args.seed}"
            .replace(".", "p")
            .replace("-", "m")
        )
        run_dir = args.run_root / run_name
        if run_dir.exists() and not args.dry_run:
            print(f"[SKIP] {run_dir} already exists.")
            continue

        cmd_train = [
            "python3",
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
            "--non-decision-target",
            str(ndt),
            "--non-decision-reg-weight",
            str(ndt_w),
            "--wfpt-loss-weight",
            str(wfpt_w),
            "--learning-rate",
            str(lr),
        ]
        run_command(cmd_train, dry_run=args.dry_run)

        cmd_eval = [
            "python3",
            "scripts/evaluate_agent.py",
            "--run",
            str(run_dir),
            "--use-best",
        ]
        run_command(cmd_eval, dry_run=args.dry_run)

        cmd_dashboard = [
            "python3",
            "scripts/make_dashboard.py",
            "--opts.agent-log",
            str(run_dir / "trials_best.ndjson"),
            "--opts.reference-log",
            str(default_ref),
            "--opts.output",
            str(run_dir / "dashboard.html"),
            "--opts.agent-name",
            f"R-DDM finetune sweep {idx}",
            "--opts.reference-name",
            "Reference",
        ]
        run_command(cmd_dashboard, dry_run=args.dry_run)

        summaries.append(
            {
                "run": run_name,
                "task": args.task,
                "non_decision_target": ndt,
                "non_decision_reg_weight": ndt_w,
                "wfpt_loss_weight": wfpt_w,
                "learning_rate": lr,
            }
        )

        run_summary = summarise_run(run_dir)
        summaries[-1].update(run_summary)

        if is_bad(run_summary):
            print(f"[FAIL] {run_dir} is degenerate. Aborting.")
            summaries[-1]["status"] = "FAILED"
            continue

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
