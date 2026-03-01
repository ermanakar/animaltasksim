#!/usr/bin/env python
"""History finetuning Phase 4 sweep.

Sweeps per_trial_history_weight x history_bias_lr x seed using the
3-phase curriculum + Phase 4 history finetuning to find settings that
improve win-stay/lose-shift without regressing psych/chrono.

Based on drift_calibration_sweep.py pattern.
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro


@dataclass(slots=True)
class HistoryFinetuneArgs:
    """History finetuning Phase 4 sweep arguments."""

    run_root: Path = Path("runs/history_finetune_sweep")
    """Root directory where sweep runs are stored."""

    per_trial_history_weights: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.8])
    """Per-trial history loss weights to sweep."""

    history_bias_lrs: list[float] = field(default_factory=lambda: [1e-3, 3e-3, 1e-2])
    """History bias learning rates to sweep."""

    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    """Random seeds to sweep."""

    drift_magnitude_target: float = 6.0
    """Drift magnitude target (calibrated value)."""

    drift_scale: float = 10.0
    """Initialization scale."""

    phase4_epochs: int = 10
    """Number of Phase 4 history finetuning epochs."""

    phase4_choice_weight: float = 1.0
    """Choice weight during Phase 4."""

    phase4_drift_magnitude_weight: float = 0.5
    """Drift magnitude weight during Phase 4."""

    episodes: int = 20
    """Training episodes per run."""

    lapse_rate: float = 0.05
    """Fixed rollout lapse rate."""

    dry_run: bool = False
    """Print commands without executing."""


def run_command(cmd: list[str], *, dry_run: bool) -> int:
    """Run a command, returning the exit code."""
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    return result.returncode


def load_json(path: Path) -> dict | None:
    """Load JSON file, returning None if missing."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarise_run(run_dir: Path) -> dict[str, object]:
    """Extract key metrics from a completed run."""
    summary: dict[str, object] = {}

    metrics = load_json(run_dir / "metrics.json")
    if metrics and "metrics" in metrics:
        data = metrics["metrics"]
        summary.update(
            {
                "psych_slope": data.get("psychometric", {}).get("slope"),
                "psych_bias": data.get("psychometric", {}).get("bias"),
                "lapse_low": data.get("psychometric", {}).get("lapse_low"),
                "lapse_high": data.get("psychometric", {}).get("lapse_high"),
                "chrono_slope": data.get("chronometric", {}).get("slope_ms_per_unit"),
                "win_stay": data.get("history", {}).get("win_stay"),
                "lose_shift": data.get("history", {}).get("lose_shift"),
                "commit_rate": data.get("basic", {}).get("commit_rate"),
                "chrono_ok": data.get("quality", {}).get("chronometric_ok"),
                "chrono_overshoot": data.get("quality", {}).get("chrono_overshoot"),
                "degenerate": data.get("quality", {}).get("degenerate"),
            }
        )
    return summary


def main(args: HistoryFinetuneArgs) -> None:
    """Run the history finetuning Phase 4 sweep."""
    args.run_root.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(
        args.per_trial_history_weights,
        args.history_bias_lrs,
        args.seeds,
    ))
    summaries: list[dict[str, object]] = []
    failures: list[str] = []

    print(f"\n{'='*70}")
    print("HISTORY FINETUNING PHASE 4 SWEEP")
    print(f"{'='*70}")
    print(f"Per-trial history weights: {args.per_trial_history_weights}")
    print(f"History bias LRs: {args.history_bias_lrs}")
    print(f"Seeds: {args.seeds}")
    print(f"Total runs: {len(combos)}")
    print(f"Drift magnitude target: {args.drift_magnitude_target}")
    print(f"Phase 4 epochs: {args.phase4_epochs}")
    print(f"Output: {args.run_root}")
    print(f"{'='*70}\n")

    for i, (pt_weight, hb_lr, seed) in enumerate(combos, 1):
        run_name = f"ptw{pt_weight}_lr{hb_lr}_seed{seed}".replace(".", "p")
        run_dir = args.run_root / run_name

        print(f"\n{'='*70}")
        print(f"[{i}/{len(combos)}] {run_name}")
        print(f"{'='*70}")

        # Train: 3-phase + Phase 4 history finetuning
        cmd_train = [
            sys.executable,
            "scripts/train_hybrid_curriculum.py",
            "--task=ibl_2afc",
            f"--output-dir={run_dir}",
            f"--seed={seed}",
            f"--drift-scale={args.drift_scale}",
            f"--drift-magnitude-target={args.drift_magnitude_target}",
            f"--lapse-rate={args.lapse_rate}",
            f"--episodes={args.episodes}",
            "--no-use-default-curriculum",
            "--no-allow-early-stopping",
            # Phase 1: RT structure only
            "--phase1-epochs=15",
            "--phase1-choice-weight=0.0",
            "--phase1-rt-weight=1.0",
            "--phase1-history-weight=0.0",
            "--phase1-drift-magnitude-weight=0.5",
            # Phase 2: Add choice
            "--phase2-epochs=10",
            "--phase2-choice-weight=0.5",
            "--phase2-rt-weight=0.8",
            "--phase2-history-weight=0.1",
            "--phase2-drift-magnitude-weight=0.5",
            # Phase 3: Full balance
            "--phase3-epochs=10",
            "--phase3-choice-weight=1.0",
            "--phase3-rt-weight=0.5",
            "--phase3-history-weight=0.2",
            "--phase3-drift-magnitude-weight=0.5",
            # Phase 4: History finetuning
            "--phase4-history-finetune",
            f"--phase4-epochs={args.phase4_epochs}",
            f"--phase4-per-trial-history-weight={pt_weight}",
            f"--phase4-choice-weight={args.phase4_choice_weight}",
            f"--phase4-drift-magnitude-weight={args.phase4_drift_magnitude_weight}",
            f"--phase4-history-bias-lr={hb_lr}",
        ]

        rc = run_command(cmd_train, dry_run=args.dry_run)
        if rc != 0:
            print(f"[WARN] Training failed for {run_name}")
            failures.append(run_name)
            continue

        # Evaluate
        cmd_eval = [
            sys.executable,
            "scripts/evaluate_agent.py",
            "--run",
            str(run_dir),
        ]
        rc = run_command(cmd_eval, dry_run=args.dry_run)
        if rc != 0:
            print(f"[WARN] Evaluation failed for {run_name}")
            failures.append(run_name)
            continue

        if not args.dry_run:
            row: dict[str, object] = {
                "run": run_name,
                "pt_history_weight": pt_weight,
                "history_bias_lr": hb_lr,
                "seed": seed,
            }
            row.update(summarise_run(run_dir))
            summaries.append(row)

            # Print key metrics inline
            ps = row.get("psych_slope")
            cs = row.get("chrono_slope")
            ws = row.get("win_stay")
            ls = row.get("lose_shift")
            print(f"\n  >> psych={ps}, chrono={cs}, win_stay={ws}, lose_shift={ls}")

    # Write CSV summary
    if summaries and not args.dry_run:
        csv_path = args.run_root / "sweep_summary.csv"
        fieldnames = list(summaries[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        print(f"\n[INFO] Summary saved to {csv_path}")

        # Print aggregated results by (pt_weight, lr)
        print(f"\n{'='*70}")
        print("AGGREGATED RESULTS (mean +/- std across seeds)")
        print(f"{'='*70}")
        print(
            f"{'pt_weight':>10} | {'lr':>10} | {'psych_slope':>14} | {'chrono_slope':>14} | "
            f"{'win_stay':>10} | {'lose_shift':>10}"
        )
        print("-" * 85)

        for pt_weight in args.per_trial_history_weights:
            for hb_lr in args.history_bias_lrs:
                rows = [
                    r for r in summaries
                    if r["pt_history_weight"] == pt_weight and r["history_bias_lr"] == hb_lr
                ]
                if not rows:
                    continue
                ps_vals = [r["psych_slope"] for r in rows if r.get("psych_slope") is not None]
                cs_vals = [r["chrono_slope"] for r in rows if r.get("chrono_slope") is not None]
                ws_vals = [r["win_stay"] for r in rows if r.get("win_stay") is not None]
                ls_vals = [r["lose_shift"] for r in rows if r.get("lose_shift") is not None]

                def _fmt(vals: list) -> str:
                    if not vals:
                        return "N/A"
                    m = statistics.mean(vals)
                    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    return f"{m:.2f} +/- {s:.2f}"

                print(
                    f"{pt_weight:>10.1f} | {hb_lr:>10.4f} | {_fmt(ps_vals):>14} | "
                    f"{_fmt(cs_vals):>14} | {_fmt(ws_vals):>10} | {_fmt(ls_vals):>10}"
                )

        print(
            "\nTargets: psych_slope ~ 13.2, chrono_slope ~ -36 ms/unit, "
            "win_stay ~ 0.724, lose_shift ~ 0.427"
        )

    if failures:
        print(f"\n[WARN] {len(failures)} runs failed: {failures}")


if __name__ == "__main__":
    main(tyro.cli(HistoryFinetuneArgs))
