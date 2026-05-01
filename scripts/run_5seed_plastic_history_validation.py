#!/usr/bin/env python3
"""Run and summarize 5-seed plastic-history validation experiments.

Usage:
    python scripts/run_5seed_plastic_history_validation.py --variant pure
    python scripts/run_5seed_plastic_history_validation.py --variant pure_v2
    python scripts/run_5seed_plastic_history_validation.py --variant pure_v3
    python scripts/run_5seed_plastic_history_validation.py --variant pure_v4
    python scripts/run_5seed_plastic_history_validation.py --variant pure_v5
    python scripts/run_5seed_plastic_history_validation.py --variant assisted
    python scripts/run_5seed_plastic_history_validation.py --variant all --summarize-only
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro


SEEDS = [42, 123, 456, 789, 1337]
REFERENCE_SUMMARY = (
    "Reference (per-session mean ± std): psych=20.0±5.7, "
    "chrono=-51±64, WS=0.72±0.08, LS=0.47±0.10"
)

COMMON_ARGS = [
    "--task=ibl_2afc",
    "--drift-scale=10.0",
    "--drift-magnitude-target=9.0",
    "--lapse-rate=0.05",
    "--episodes=20",
    "--max-sessions=80",
    "--max-trials-per-session=128",
    "--history-bias-scale=2.0",
    "--history-drift-scale=0.3",
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
    "--phase4-epochs=8",
    "--phase4-per-trial-history-weight=0.5",
    "--phase4-choice-weight=1.0",
    "--phase4-drift-magnitude-weight=0.1",
]

VARIANT_CONFIG = {
    "assisted": {
        "prefix": "plastic_history_seed_",
        "label": "Assisted Plastic History",
        "extra_args": [
            "--inject-win-tendency=0.30",
            "--inject-lose-tendency=0.15",
            "--anneal-history-injection",
            "--history-injection-alpha-start=1.0",
            "--history-injection-alpha-end=0.0",
            "--phase4-history-distillation-weight=1.0",
        ],
    },
    "pure": {
        "prefix": "plastic_history_pure_seed_",
        "label": "Pure Plastic History",
        "extra_args": [
            "--phase4-history-distillation-weight=0.0",
        ],
    },
    "pure_v2": {
        "prefix": "plastic_history_pure_v2_seed_",
        "label": "Pure Plastic History V2",
        "extra_args": [
            "--phase4-history-distillation-weight=0.0",
            "--phase4-epochs=16",
            "--phase4-per-trial-history-weight=1.0",
            "--phase4-choice-weight=0.5",
        ],
    },
    "pure_v3": {
        "prefix": "plastic_history_pure_v3_seed_",
        "label": "Pure Plastic History V3",
        "extra_args": [
            "--phase4-history-distillation-weight=0.0",
        ],
    },
    "pure_v4": {
        "prefix": "plastic_history_pure_v4_seed_",
        "label": "Pure Plastic History V4",
        "extra_args": [
            "--phase4-history-distillation-weight=0.0",
        ],
    },
    "pure_v5": {
        "prefix": "plastic_history_pure_v5_seed_",
        "label": "Pure Plastic History V5",
        "extra_args": [
            "--phase4-history-distillation-weight=0.0",
        ],
    },
}


@dataclass(slots=True)
class PlasticHistoryValidationArgs:
    """Arguments for plastic-history validation runs."""

    variant: Literal["assisted", "pure", "pure_v2", "pure_v3", "pure_v4", "pure_v5", "both", "all"] = "all"
    summarize_only: bool = False


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _run_dir(variant: str, seed: int) -> Path:
    return Path("runs") / f"{VARIANT_CONFIG[variant]['prefix']}{seed}"


def run_seed(seed: int, variant: str) -> Path:
    run_dir = _run_dir(variant, seed)
    print(f"\n{'=' * 70}")
    print(f"  {VARIANT_CONFIG[variant]['label']} | Seed {seed}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable,
        "scripts/train_hybrid_curriculum.py",
        f"--output-dir={run_dir}",
        f"--seed={seed}",
        *COMMON_ARGS,
        *VARIANT_CONFIG[variant]["extra_args"],
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(
        [sys.executable, "scripts/evaluate_agent.py", "--run", str(run_dir)],
        check=True,
    )
    return run_dir


def _extract_metrics(seed: int, variant: str) -> dict[str, float | int | None]:
    run_dir = _run_dir(variant, seed)
    metrics_raw = _load_json(run_dir / "metrics.json")
    metrics = metrics_raw.get("metrics", metrics_raw)
    training_metrics_raw = _load_json(run_dir / "training_metrics.json")
    training_metrics = training_metrics_raw.get("training", training_metrics_raw)

    teacher_alpha = training_metrics.get("history_teacher_alpha", [])
    distill_loss = training_metrics.get("epoch_history_distillation", [])
    win_tendency = training_metrics.get("mean_win_stay_tendency", [])
    lose_tendency = training_metrics.get("mean_lose_stay_tendency", [])
    plastic_tendency = training_metrics.get("mean_plastic_stay_tendency", [])
    reward_prediction = training_metrics.get("epoch_reward_prediction", [])

    return {
        "seed": seed,
        "psych": metrics.get("psychometric", {}).get("slope"),
        "chrono": metrics.get("chronometric", {}).get("slope_ms_per_unit"),
        "ws": metrics.get("history", {}).get("win_stay"),
        "ls": metrics.get("history", {}).get("lose_shift"),
        "lapse": metrics.get("psychometric", {}).get("lapse_low"),
        "bias": metrics.get("psychometric", {}).get("bias"),
        "sticky": metrics.get("history", {}).get("sticky_choice"),
        "teacher_alpha_final": teacher_alpha[-1] if teacher_alpha else None,
        "distill_loss_final": distill_loss[-1] if distill_loss else None,
        "win_tendency_final": win_tendency[-1] if win_tendency else None,
        "lose_tendency_final": lose_tendency[-1] if lose_tendency else None,
        "plastic_tendency_final": plastic_tendency[-1] if plastic_tendency else None,
        "reward_prediction_final": reward_prediction[-1] if reward_prediction else None,
    }


def summarize_variant(variant: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {VARIANT_CONFIG[variant]['label']} Summary")
    print(f"{'=' * 70}")

    rows: list[dict[str, float | int | None]] = []
    for seed in SEEDS:
        run_dir = _run_dir(variant, seed)
        metrics_path = run_dir / "metrics.json"
        training_metrics_path = run_dir / "training_metrics.json"
        if not metrics_path.exists() or not training_metrics_path.exists():
            print(f"  seed={seed}: MISSING")
            continue
        row = _extract_metrics(seed, variant)
        rows.append(row)
        print(
            f"  seed={seed:>5} | psych={float(row['psych']):5.2f}"
            f" | chrono={float(row['chrono']):6.1f} | WS={float(row['ws']):.3f}"
            f" | LS={float(row['ls']):.3f} | plastic_t={float(row['plastic_tendency_final'] or 0.0):+.3f}"
            f" | reward_pred={float(row['reward_prediction_final'] or 0.0):.3f}"
        )

    if len(rows) < 1:
        return

    print(f"\n  {'─' * 66}")
    for key, label in [
        ("psych", "Psych slope"),
        ("chrono", "Chrono slope"),
        ("ws", "Win-stay"),
        ("ls", "Lose-shift"),
        ("sticky", "Sticky choice"),
        ("plastic_tendency_final", "Plastic tendency"),
        ("reward_prediction_final", "Reward pred loss"),
    ]:
        vals = [float(row[key]) for row in rows if row[key] is not None]
        if not vals:
            continue
        if len(vals) == 1:
            print(f"  {label:<16} = {vals[0]:7.3f}")
        else:
            print(f"  {label:<16} = {_mean(vals):7.3f} ± {_std(vals):.3f}")

    print(f"\n  {REFERENCE_SUMMARY}")


def main(args: PlasticHistoryValidationArgs) -> None:
    """Run or summarize the requested plastic-history validation family."""
    if args.variant in {"both", "all"}:
        variants = ["assisted", "pure", "pure_v2", "pure_v3", "pure_v4", "pure_v5"]
    else:
        variants = [args.variant]

    if not args.summarize_only:
        for variant in variants:
            for seed in SEEDS:
                run_seed(seed, variant)

    for variant in variants:
        summarize_variant(variant)


if __name__ == "__main__":
    main(tyro.cli(PlasticHistoryValidationArgs))
