"""Multi-seed validation sweep for the IBL drift-rate bias experiment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SEEDS = [42, 123, 256, 789, 1337]

# Exact v6 config (from runs/ibl_drift_v6_max/config.json)
BASE_ARGS = [
    sys.executable,
    "scripts/train_hybrid_curriculum.py",
    "--task", "ibl_2afc",
    "--episodes", "30",
    "--max-sessions", "80",
    "--history-phase-epochs", "20",
    "--history-history-supervision-weight", "0.8",
    "--history-per-trial-history-weight", "1.5",
    "--history-bias-scale", "1.0",
    "--history-drift-scale", "15.0",
]


def _extract_metrics(seed: int, metrics: dict[str, object]) -> dict[str, object]:
    """Extract key metrics into a flat dict."""
    return {
        "seed": seed,
        "status": "ok",
        "win_stay": metrics.get("history", {}).get("win_stay"),
        "lose_shift": metrics.get("history", {}).get("lose_shift"),
        "sticky_choice": metrics.get("history", {}).get("sticky_choice"),
        "psych_slope": metrics.get("psychometric", {}).get("slope"),
        "chrono_slope": metrics.get("chronometric", {}).get("slope_ms_per_unit"),
        "bias": metrics.get("psychometric", {}).get("bias"),
    }


def run_seed(seed: int, output_dir: Path) -> dict[str, object]:
    """Train + evaluate a single seed."""
    print(f"\n{'='*80}")
    print(f"SEED {seed}")
    print(f"{'='*80}\n")

    run_dir = output_dir / f"seed_{seed}"

    # Train
    cmd = [*BASE_ARGS, "--seed", str(seed), "--output-dir", str(run_dir)]
    print(f"Training: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: seed {seed} training returned {result.returncode}")
        return {"seed": seed, "status": "train_failed"}

    # Evaluate
    eval_cmd = [sys.executable, "scripts/evaluate_agent.py", "--run", str(run_dir)]
    print(f"\nEvaluating: {' '.join(eval_cmd)}")
    result = subprocess.run(eval_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: seed {seed} evaluation returned {result.returncode}")
        return {"seed": seed, "status": "eval_failed"}

    # Read metrics
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {"seed": seed, "status": "no_metrics"}

    with open(metrics_path) as f:
        raw = json.load(f)

    # Handle nested "metrics" key from evaluate_agent.py
    metrics = raw.get("metrics", raw)

    return _extract_metrics(seed, metrics)


def summarize(results: list[dict[str, object]]) -> None:
    """Print summary statistics."""
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    print(f"\n{'='*80}")
    print("MULTI-SEED VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Seeds run: {len(results)}, succeeded: {len(ok)}, failed: {len(failed)}")

    if failed:
        print(f"Failed seeds: {[r['seed'] for r in failed]}")

    if not ok:
        print("No successful runs to summarize.")
        return

    # Per-seed table
    metrics = ["win_stay", "lose_shift", "sticky_choice", "psych_slope", "chrono_slope"]
    targets = {"win_stay": 0.724, "lose_shift": 0.427, "sticky_choice": None, "psych_slope": 13.2, "chrono_slope": None}

    print(f"\n{'Seed':<8}", end="")
    for m in metrics:
        print(f"{m:<16}", end="")
    print()
    print("-" * 88)

    for r in ok:
        print(f"{r['seed']:<8}", end="")
        for m in metrics:
            val = r.get(m)
            if val is not None:
                print(f"{val:<16.4f}", end="")
            else:
                print(f"{'N/A':<16}", end="")
        print()

    # Mean ± std
    import numpy as np

    print(f"\n{'-'*88}")
    print(f"{'Mean':<8}", end="")
    for m in metrics:
        vals = [r[m] for r in ok if r.get(m) is not None]
        if vals:
            print(f"{np.mean(vals):<16.4f}", end="")
        else:
            print(f"{'N/A':<16}", end="")
    print()

    print(f"{'Std':<8}", end="")
    for m in metrics:
        vals = [r[m] for r in ok if r.get(m) is not None]
        if vals:
            print(f"{np.std(vals):<16.4f}", end="")
        else:
            print(f"{'N/A':<16}", end="")
    print()

    if any(v is not None for v in targets.values()):
        print(f"\n{'Target':<8}", end="")
        for m in metrics:
            t = targets.get(m)
            if t is not None:
                print(f"{t:<16.4f}", end="")
            else:
                print(f"{'—':<16}", end="")
        print()

    print(f"\n{'='*80}\n")


def main() -> None:
    """Run the sweep."""
    output_dir = Path("runs/seed_sweep_ibl_v6")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if seed 42 already exists
    existing_v6 = Path("runs/ibl_drift_v6_max")
    seed42_dir = output_dir / "seed_42"
    if existing_v6.exists() and not seed42_dir.exists():
        print("Linking existing v6 run as seed_42...")
        import shutil
        shutil.copytree(existing_v6, seed42_dir)

    results = []
    for seed in SEEDS:
        run_dir = output_dir / f"seed_{seed}"
        if (run_dir / "metrics.json").exists():
            print(f"\nSeed {seed} already has metrics, loading...")
            with open(run_dir / "metrics.json") as f:
                raw = json.load(f)
            metrics = raw.get("metrics", raw)
            results.append(_extract_metrics(seed, metrics))
        else:
            results.append(run_seed(seed, output_dir))

    # Save results
    summary_path = output_dir / "sweep_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    summarize(results)


if __name__ == "__main__":
    main()
