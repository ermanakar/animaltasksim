"""One-command pipeline: train → evaluate → compare to animal reference.

Usage:
    python scripts/make_compare.py --env ibl_2afc --agent sticky_q --episodes 3
    python scripts/make_compare.py --env rdm --agent ppo --episodes 2 --seed 99
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro


@dataclass(slots=True)
class CompareArgs:
    """Train an agent and compare it to animal reference data."""

    env: Literal["ibl_2afc", "rdm"] = "ibl_2afc"
    """Task environment."""

    agent: Literal["sticky_q", "bayes", "ppo"] = "sticky_q"
    """Agent architecture."""

    episodes: int = 3
    """Number of training episodes."""

    trials_per_episode: int = 400
    """Trials per episode."""

    seed: int = 42
    """Random seed."""

    out: Path | None = None
    """Output directory (auto-generated if not set)."""


def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess, printing its output and raising on failure."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main(args: CompareArgs) -> None:
    python = sys.executable
    run_dir = args.out or Path(f"runs/compare_{args.env}_{args.agent}_s{args.seed}")

    # Step 1: Train
    train_cmd = [
        python, "scripts/train_agent.py",
        "--env", args.env,
        "--agent", args.agent,
        "--episodes", str(args.episodes),
        "--trials-per-episode", str(args.trials_per_episode),
        "--seed", str(args.seed),
        "--out", str(run_dir),
    ]
    _run(train_cmd, f"Step 1/3: Train {args.agent} on {args.env}")

    # Step 2: Evaluate
    eval_cmd = [
        python, "scripts/evaluate_agent.py",
        "--run", str(run_dir),
    ]
    _run(eval_cmd, "Step 2/3: Evaluate behavioural metrics")

    # Step 3: Compare (generate leaderboard including the new run)
    compare_cmd = [
        python, "scripts/compare_runs.py",
    ]
    _run(compare_cmd, "Step 3/3: Generate leaderboard comparison")

    # Summary
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print(f"\n{'='*60}")
        print(f"  Results: {run_dir}")
        print(f"{'='*60}")

        m = metrics.get("metrics", metrics)
        psych = m.get("psychometric", {})
        chrono = m.get("chronometric", {})
        hist = m.get("history", {})
        flags = m.get("quality_flags", {})

        print(f"  Psychometric slope: {psych.get('slope', 'N/A')}")
        print(f"  Chronometric slope: {chrono.get('slope_ms_per_unit', 'N/A')}")
        if chrono.get("corrected_slope") is not None:
            print(f"  Corrected slope:    {chrono['corrected_slope']}")
        print(f"  Ceiling fraction:   {chrono.get('ceiling_fraction', 'N/A')}")
        print(f"  Win-stay:           {hist.get('win_stay', 'N/A')}")
        print(f"  Lose-shift:         {hist.get('lose_shift', 'N/A')}")
        print(f"  Bias:               {psych.get('bias', 'N/A')}")
        print(f"  Quality flags:      {flags}")
        print(f"\n  Leaderboard:        runs/leaderboard.html")
        print(f"  Metrics:            {metrics_path}")
    else:
        print(f"\nWarning: metrics.json not found at {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main(tyro.cli(CompareArgs))
