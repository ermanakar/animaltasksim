#!/usr/bin/env python3
"""5-seed validation of best co-evolution config (win=0.30, lose=0.15, target=9).

Usage:
    python scripts/run_5seed_validation.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SEEDS = [42, 123, 456, 789, 1337]
WIN_T = 0.30
LOSE_T = 0.15
OUTPUT_ROOT = Path("runs/coevolution_5seed_validation")

COMMON_ARGS = [
    "--task=ibl_2afc",
    "--drift-scale=10.0",
    "--drift-magnitude-target=9.0",
    "--lapse-rate=0.05",
    "--episodes=20",
    "--history-bias-scale=2.0",
    "--history-drift-scale=0.3",
    f"--inject-win-tendency={WIN_T}",
    f"--inject-lose-tendency={LOSE_T}",
    "--no-use-default-curriculum",
    "--no-allow-early-stopping",
    "--phase1-epochs=15",
    "--phase1-choice-weight=0.0",
    "--phase1-rt-weight=1.0",
    "--phase1-history-weight=0.0",
    "--phase1-drift-magnitude-weight=0.5",
    "--phase2-epochs=10",
    "--phase2-choice-weight=0.5",
    "--phase2-rt-weight=0.8",
    "--phase2-history-weight=0.1",
    "--phase2-drift-magnitude-weight=0.5",
    "--phase3-epochs=10",
    "--phase3-choice-weight=1.0",
    "--phase3-rt-weight=0.5",
    "--phase3-history-weight=0.2",
    "--phase3-drift-magnitude-weight=0.5",
]


def run_seed(seed: int) -> Path:
    """Train + evaluate a single seed. Returns the run directory."""
    run_dir = OUTPUT_ROOT / f"seed{seed}"
    print(f"\n{'=' * 70}")
    print(f"  Seed {seed}")
    print(f"{'=' * 70}")

    # Train
    cmd = [
        sys.executable, "scripts/train_hybrid_curriculum.py",
        f"--output-dir={run_dir}",
        f"--seed={seed}",
        *COMMON_ARGS,
    ]
    subprocess.run(cmd, check=True)

    # Evaluate
    subprocess.run(
        [sys.executable, "scripts/evaluate_agent.py", "--run", str(run_dir)],
        check=True,
    )
    return run_dir


def summarize() -> None:
    """Print summary table from all completed seeds."""
    print(f"\n{'=' * 70}")
    print("  5-SEED VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Config: win_t={WIN_T}, lose_t={LOSE_T}, drift_magnitude_target=9.0")
    print(f"{'=' * 70}\n")

    rows: list[dict] = []
    for seed in SEEDS:
        metrics_path = OUTPUT_ROOT / f"seed{seed}" / "metrics.json"
        if not metrics_path.exists():
            print(f"  seed={seed}: MISSING")
            continue
        m = json.loads(metrics_path.read_text())["metrics"]
        row = {
            "seed": seed,
            "psych": m["psychometric"]["slope"],
            "chrono": m["chronometric"]["slope_ms_per_unit"],
            "ws": m["history"]["win_stay"],
            "ls": m["history"]["lose_shift"],
            "lapse": m["psychometric"]["lapse_low"],
            "bias": m["psychometric"]["bias"],
        }
        rows.append(row)
        print(
            f"  seed={seed:>5} | psych={row['psych']:5.2f} | chrono={row['chrono']:6.1f}"
            f" | WS={row['ws']:.3f} | LS={row['ls']:.3f}"
            f" | lapse={row['lapse']:.3f} | bias={row['bias']:+.3f}"
        )

    if len(rows) >= 2:
        print(f"\n  {'─' * 66}")
        for key, label in [
            ("psych", "Psych slope"),
            ("chrono", "Chrono slope"),
            ("ws", "Win-stay"),
            ("ls", "Lose-shift"),
            ("lapse", "Lapse"),
        ]:
            vals = [r[key] for r in rows]
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"  {label:<14} = {mean:7.3f} ± {sd:.3f}")

    print("\n  Reference (per-session mean ± std): psych=20.0±5.7, chrono=-51±64 (lit. -36), WS=0.72±0.08, LS=0.47±0.10")


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        run_seed(seed)

    summarize()
