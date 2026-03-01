#!/usr/bin/env python
"""History scale hyperparameter sweep.

Sweeps history_bias_scale x history_drift_scale as FROZEN (non-trainable)
parameters using the 3-phase curriculum to find values that produce
correct win-stay (target 0.724) and lose-shift (target 0.427) without
degrading psychometric slope (target ~13.2) or chronometric slope (target -36).

Key insight: Both scale params are normally nn.Parameters that get shrunk
by the optimizer during training (lesson 16). This sweep freezes them
and treats them as pure hyperparameters, letting the history networks
learn appropriate stay_tendency values against fixed scales.

Phase 1 (discovery): 1 seed, broad grid -> ~16 runs (~5 hours)
Phase 2 (validation): 3 seeds around best combos from Phase 1
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
class HistoryScaleArgs:
    """History scale hyperparameter sweep arguments."""

    run_root: Path = Path("runs/history_scale_sweep")
    """Root directory where sweep runs are stored."""

    history_bias_scales: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 4.0, 8.0]
    )
    """history_bias_scale values to sweep (floor clamp at 1.0)."""

    history_drift_scales: list[float] = field(
        default_factory=lambda: [0.0, 0.3, 1.0, 2.0]
    )
    """history_drift_scale values to sweep (no floor)."""

    seeds: list[int] = field(default_factory=lambda: [42])
    """Random seeds to sweep. Phase 1: [42]. Phase 2: [42, 123, 456]."""

    drift_magnitude_target: float = 6.0
    """Drift magnitude target (calibrated value from drift_calibration_v2)."""

    drift_scale: float = 10.0
    """Initialization scale for drift_head."""

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
                "chrono_r2": data.get("chronometric", {}).get("r_squared"),
                "win_stay": data.get("history", {}).get("win_stay"),
                "lose_shift": data.get("history", {}).get("lose_shift"),
                "sticky_choice": data.get("history", {}).get("sticky_choice"),
                "commit_rate": data.get("basic", {}).get("commit_rate"),
                "chrono_ok": data.get("quality", {}).get("chronometric_ok"),
                "chrono_overshoot": data.get("quality", {}).get("chrono_overshoot"),
                "degenerate": data.get("quality", {}).get("degenerate"),
            }
        )
    return summary


def main(args: HistoryScaleArgs) -> None:
    """Run the history scale hyperparameter sweep."""
    args.run_root.mkdir(parents=True, exist_ok=True)

    combos = list(
        itertools.product(
            args.history_bias_scales,
            args.history_drift_scales,
            args.seeds,
        )
    )
    summaries: list[dict[str, object]] = []
    failures: list[str] = []

    print(f"\n{'='*70}")
    print("HISTORY SCALE HYPERPARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"history_bias_scales: {args.history_bias_scales}")
    print(f"history_drift_scales: {args.history_drift_scales}")
    print(f"Seeds: {args.seeds}")
    print(f"Total runs: {len(combos)}")
    print(f"Drift magnitude target: {args.drift_magnitude_target}")
    print(f"Lapse rate: {args.lapse_rate}")
    print(f"Output: {args.run_root}")
    print(f"{'='*70}\n")

    for i, (hb_scale, hd_scale, seed) in enumerate(combos, 1):
        run_name = f"hb{hb_scale}_hd{hd_scale}_seed{seed}".replace(".", "p")
        run_dir = args.run_root / run_name

        print(f"\n{'='*70}")
        print(f"[{i}/{len(combos)}] {run_name}")
        print(
            f"  history_bias_scale={hb_scale}, "
            f"history_drift_scale={hd_scale}, seed={seed}"
        )
        print(f"{'='*70}")

        # Train: 3-phase curriculum with frozen history scales
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
            f"--history-bias-scale={hb_scale}",
            f"--history-drift-scale={hd_scale}",
            "--freeze-history-scales",
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
                "history_bias_scale": hb_scale,
                "history_drift_scale": hd_scale,
                "seed": seed,
            }
            row.update(summarise_run(run_dir))
            summaries.append(row)

            # Print key metrics inline
            ps = row.get("psych_slope")
            cs = row.get("chrono_slope")
            ws = row.get("win_stay")
            ls = row.get("lose_shift")
            print(
                f"\n  >> psych={ps}, chrono={cs}, "
                f"win_stay={ws}, lose_shift={ls}"
            )

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

        # Print aggregated results by (hb_scale, hd_scale)
        print(f"\n{'='*70}")
        print("AGGREGATED RESULTS (mean +/- std across seeds)")
        print(f"{'='*70}")
        print(
            f"{'hb_scale':>10} | {'hd_scale':>10} | {'psych_slope':>14} | "
            f"{'chrono_slope':>14} | {'win_stay':>10} | {'lose_shift':>10}"
        )
        print("-" * 85)

        for hb_scale in args.history_bias_scales:
            for hd_scale in args.history_drift_scales:
                rows = [
                    r
                    for r in summaries
                    if r["history_bias_scale"] == hb_scale
                    and r["history_drift_scale"] == hd_scale
                ]
                if not rows:
                    continue

                ps_vals = [
                    r["psych_slope"]
                    for r in rows
                    if r.get("psych_slope") is not None
                ]
                cs_vals = [
                    r["chrono_slope"]
                    for r in rows
                    if r.get("chrono_slope") is not None
                ]
                ws_vals = [
                    r["win_stay"]
                    for r in rows
                    if r.get("win_stay") is not None
                ]
                ls_vals = [
                    r["lose_shift"]
                    for r in rows
                    if r.get("lose_shift") is not None
                ]

                def _fmt(vals: list) -> str:
                    if not vals:
                        return "N/A"
                    m = statistics.mean(vals)
                    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    return f"{m:.2f} +/- {s:.2f}"

                print(
                    f"{hb_scale:>10.1f} | {hd_scale:>10.1f} | "
                    f"{_fmt(ps_vals):>14} | {_fmt(cs_vals):>14} | "
                    f"{_fmt(ws_vals):>10} | {_fmt(ls_vals):>10}"
                )

        print(
            "\nTargets: psych_slope ~ 13.2, chrono_slope ~ -36 ms/unit, "
            "win_stay ~ 0.724, lose_shift ~ 0.427"
        )

    if failures:
        print(f"\n[WARN] {len(failures)} runs failed: {failures}")


if __name__ == "__main__":
    main(tyro.cli(HistoryScaleArgs))
