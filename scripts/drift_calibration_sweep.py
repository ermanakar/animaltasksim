#!/usr/bin/env python
"""Drift magnitude target calibration sweep.

Sweeps drift_magnitude_target × seed using the 3-phase curriculum
to find the target that produces psych slope nearest IBL (13.2).

Previous finding: drift_scale (init only) had NO effect because the
drift_magnitude loss pulls drift_gain to a fixed target during training.
This sweep varies that target directly.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from scripts._sweep_utils import (
    format_mean_std,
    run_command,
    slug,
    summarise_behavior_metrics,
    write_csv,
)


@dataclass(slots=True)
class CalibrationArgs:
    """Drift magnitude target calibration arguments."""

    run_root: Path = Path("runs/drift_calibration_v2")
    """Root directory where sweep runs are stored."""

    drift_targets: list[float] = field(default_factory=lambda: [6.0, 7.0, 8.0, 9.0])
    """Drift magnitude target values to sweep (regularization pulls drift_gain here)."""

    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    """Random seeds to sweep."""

    drift_scale: float = 10.0
    """Initialization scale (less important now that target controls final value)."""

    choice_weight: float = 1.0
    """Fixed choice weight for all runs."""

    episodes: int = 20
    """Training episodes per run."""

    lapse_rate: float = 0.05
    """Fixed rollout lapse rate."""

    dry_run: bool = False
    """Print commands without executing."""


def main(args: CalibrationArgs) -> None:
    """Run the drift magnitude target calibration sweep."""
    args.run_root.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.drift_targets, args.seeds))
    summaries: list[dict[str, object]] = []
    failures: list[str] = []

    print(f"\n{'='*70}")
    print("DRIFT MAGNITUDE TARGET CALIBRATION SWEEP")
    print(f"{'='*70}")
    print(f"Drift targets: {args.drift_targets}")
    print(f"Seeds: {args.seeds}")
    print(f"Total runs: {len(combos)}")
    print(f"Choice weight: {args.choice_weight}")
    print(f"Lapse rate: {args.lapse_rate}")
    print(f"Output: {args.run_root}")
    print(f"{'='*70}\n")

    for i, (drift_target, seed) in enumerate(combos, 1):
        run_name = f"target{slug(drift_target)}_seed{seed}"
        run_dir = args.run_root / run_name

        print(f"\n{'='*70}")
        print(f"[{i}/{len(combos)}] {run_name}")
        print(f"{'='*70}")

        # Train
        cmd_train = [
            sys.executable,
            "scripts/train_hybrid_curriculum.py",
            "--task=ibl_2afc",
            f"--output-dir={run_dir}",
            f"--seed={seed}",
            f"--drift-scale={args.drift_scale}",
            f"--drift-magnitude-target={drift_target}",
            f"--lapse-rate={args.lapse_rate}",
            "--no-use-default-curriculum",
            "--no-allow-early-stopping",
            "--phase1-epochs=15",
            "--phase1-choice-weight=0.0",
            "--phase1-rt-weight=1.0",
            "--phase1-history-weight=0.0",
            "--phase1-drift-magnitude-weight=0.5",
            "--phase2-epochs=10",
            f"--phase2-choice-weight={args.choice_weight * 0.5}",
            "--phase2-rt-weight=0.8",
            "--phase2-history-weight=0.1",
            "--phase2-drift-magnitude-weight=0.5",
            "--phase3-epochs=10",
            f"--phase3-choice-weight={args.choice_weight}",
            "--phase3-rt-weight=0.5",
            "--phase3-history-weight=0.2",
            "--phase3-drift-magnitude-weight=0.5",
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
                "drift_target": drift_target,
                "seed": seed,
            }
            row.update(summarise_behavior_metrics(run_dir))
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
        write_csv(csv_path, summaries)
        print(f"\n[INFO] Summary saved to {csv_path}")

        # Print aggregated results by drift_target
        print(f"\n{'='*70}")
        print("AGGREGATED RESULTS (mean +/- std across seeds)")
        print(f"{'='*70}")
        print(f"{'target':>8} | {'psych_slope':>14} | {'chrono_slope':>14} | {'win_stay':>10} | {'lose_shift':>10}")
        print("-" * 70)

        for target in args.drift_targets:
            rows = [r for r in summaries if r["drift_target"] == target]
            if not rows:
                continue
            ps_vals = [r["psych_slope"] for r in rows if r.get("psych_slope") is not None]
            cs_vals = [r["chrono_slope"] for r in rows if r.get("chrono_slope") is not None]
            ws_vals = [r["win_stay"] for r in rows if r.get("win_stay") is not None]
            ls_vals = [r["lose_shift"] for r in rows if r.get("lose_shift") is not None]

            print(
                f"{target:>8.1f} | {format_mean_std(ps_vals):>14} | "
                f"{format_mean_std(cs_vals):>14} | {format_mean_std(ws_vals):>10} | "
                f"{format_mean_std(ls_vals):>10}"
            )

        print("\nTarget: psych_slope ~ 13.2, chrono_slope < 0, win_stay ~ 0.724, lose_shift ~ 0.427")

    if failures:
        print(f"\n[WARN] {len(failures)} runs failed: {failures}")


if __name__ == "__main__":
    main(tyro.cli(CalibrationArgs))
