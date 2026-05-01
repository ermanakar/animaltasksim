#!/usr/bin/env python3
"""Multi-seed validation of annealed history teacher forcing on IBL 2AFC.

Usage:
    python scripts/run_5seed_teacher_forcing_validation.py

This mirrors the existing fixed-injection 5-seed validation, but adds:
  - annealed teacher forcing from injected history to learned history
  - an explicit Phase 4 history finetuning stage
  - summary of learned win/lose tendency outputs from training_metrics.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SEEDS = [42, 123, 456, 789, 1337]
WIN_T = 0.30
LOSE_T = 0.15
OUTPUT_ROOT = Path("runs/teacher_forcing_distill_5seed_validation")

COMMON_ARGS = [
    "--task=ibl_2afc",
    "--drift-scale=10.0",
    "--drift-magnitude-target=9.0",
    "--lapse-rate=0.05",
    "--episodes=20",
    "--max-sessions=80",
    "--history-bias-scale=2.0",
    "--history-drift-scale=0.3",
    f"--inject-win-tendency={WIN_T}",
    f"--inject-lose-tendency={LOSE_T}",
    "--anneal-history-injection",
    "--history-injection-alpha-start=1.0",
    "--history-injection-alpha-end=0.0",
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
    "--phase4-history-finetune",
    "--phase4-epochs=10",
    "--phase4-per-trial-history-weight=0.5",
    "--phase4-history-distillation-weight=1.0",
    "--phase4-choice-weight=1.0",
    "--phase4-drift-magnitude-weight=0.5",
]


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _extract_metrics(seed: int, run_dir: Path) -> dict[str, float | int | None]:
    metrics_raw = _load_json(run_dir / "metrics.json")
    metrics = metrics_raw.get("metrics", metrics_raw)
    training_metrics_raw = _load_json(run_dir / "training_metrics.json")
    training_metrics = training_metrics_raw.get("training", training_metrics_raw)

    teacher_alphas = training_metrics.get("history_teacher_alpha", [])
    distill_loss = training_metrics.get("epoch_history_distillation", [])
    win_tendency = training_metrics.get("mean_win_stay_tendency", [])
    lose_tendency = training_metrics.get("mean_lose_stay_tendency", [])

    return {
        "seed": seed,
        "psych": metrics.get("psychometric", {}).get("slope"),
        "chrono": metrics.get("chronometric", {}).get("slope_ms_per_unit"),
        "ws": metrics.get("history", {}).get("win_stay"),
        "ls": metrics.get("history", {}).get("lose_shift"),
        "lapse": metrics.get("psychometric", {}).get("lapse_low"),
        "bias": metrics.get("psychometric", {}).get("bias"),
        "teacher_alpha_final": teacher_alphas[-1] if teacher_alphas else None,
        "distill_loss_final": distill_loss[-1] if distill_loss else None,
        "win_tendency_final": win_tendency[-1] if win_tendency else None,
        "lose_tendency_final": lose_tendency[-1] if lose_tendency else None,
    }


def run_seed(seed: int) -> Path:
    run_dir = OUTPUT_ROOT / f"seed{seed}"
    print(f"\n{'=' * 70}")
    print(f"  Seed {seed}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable,
        "scripts/train_hybrid_curriculum.py",
        f"--output-dir={run_dir}",
        f"--seed={seed}",
        *COMMON_ARGS,
    ]
    subprocess.run(cmd, check=True)

    subprocess.run(
        [sys.executable, "scripts/evaluate_agent.py", "--run", str(run_dir)],
        check=True,
    )
    return run_dir


def summarize() -> None:
    print(f"\n{'=' * 70}")
    print("  5-SEED ANNEALED TEACHER-FORCING VALIDATION")
    print(f"{'=' * 70}")
    print(
        "  Config: win_t=0.30, lose_t=0.15, drift_magnitude_target=9.0, "
        "teacher forcing 1.0→0.0, phase4 distillation=1.0"
    )
    print(f"{'=' * 70}\n")

    rows: list[dict[str, float | int | None]] = []
    for seed in SEEDS:
        run_dir = OUTPUT_ROOT / f"seed{seed}"
        metrics_path = run_dir / "metrics.json"
        training_metrics_path = run_dir / "training_metrics.json"
        if not metrics_path.exists() or not training_metrics_path.exists():
            print(f"  seed={seed}: MISSING")
            continue
        row = _extract_metrics(seed, run_dir)
        rows.append(row)
        print(
            f"  seed={seed:>5} | psych={row['psych']:5.2f} | chrono={row['chrono']:6.1f}"
            f" | WS={row['ws']:.3f} | LS={row['ls']:.3f}"
            f" | win_t={row['win_tendency_final']:+.3f} | lose_t={row['lose_tendency_final']:+.3f}"
            f" | distill={row['distill_loss_final']:.3f} | alpha_final={row['teacher_alpha_final']:.3f}"
        )

    if len(rows) < 2:
        return

    print(f"\n  {'─' * 66}")
    for key, label in [
        ("psych", "Psych slope"),
        ("chrono", "Chrono slope"),
        ("ws", "Win-stay"),
        ("ls", "Lose-shift"),
        ("win_tendency_final", "Win tendency"),
        ("lose_tendency_final", "Lose tendency"),
    ]:
        vals = [float(r[key]) for r in rows if r[key] is not None]
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<14} = {mean:7.3f} ± {sd:.3f}")

    print(
        "\n  Reference (per-session mean ± std): psych=20.0±5.7, chrono=-51±64, "
        "WS=0.72±0.08, LS=0.47±0.10"
    )


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        run_seed(seed)

    summarize()
