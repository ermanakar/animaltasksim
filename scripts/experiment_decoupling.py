#!/usr/bin/env python
"""Decoupling experiment: does per_trial_history_loss solve the core scientific gap?

Runs R-DDM experiments A–D on IBL 2AFC, then optionally Hybrid experiment E on
macaque RDM.  Each experiment trains, rolls out, evaluates, and prints metrics.
A final comparison table reveals whether per-trial history lifts win-stay /
lose-shift without regressing chronometric slope.

Usage:
    python scripts/experiment_decoupling.py                    # R-DDM only (fast)
    python scripts/experiment_decoupling.py --include-hybrid   # + Hybrid (~15 min)
    python scripts/experiment_decoupling.py --quick            # smoke test (fewer epochs)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class ExperimentSpec:
    """One row in the experiment matrix."""

    name: str
    agent: str  # "rddm" or "hybrid"
    task: str
    run_dir: Path
    train_cmd: list[str]
    description: str


def _rddm_spec(
    name: str,
    description: str,
    *,
    wfpt: float,
    history: float,
    per_trial: float,
    epochs: int = 30,
    seed: int = 42,
) -> ExperimentSpec:
    run_dir = ROOT / "runs" / f"decoupling_{name}"
    cmd = [
        PYTHON, str(ROOT / "scripts" / "train_r_ddm.py"),
        "--run-dir", str(run_dir),
        "--task", "ibl_2afc",
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--wfpt-loss-weight", str(wfpt),
        "--history-loss-weight", str(history),
        "--per-trial-history-weight", str(per_trial),
        "--rollout-trials", "1200",
        "--motor-delay-ms", "200",
        "--enable-history-epoch", "8",
        "--history-ramp-epochs", "6",
    ]
    return ExperimentSpec(
        name=name,
        agent="rddm",
        task="ibl_2afc",
        run_dir=run_dir,
        train_cmd=cmd,
        description=description,
    )


def _hybrid_spec(
    name: str,
    description: str,
    *,
    per_trial_history: float = 0.5,
    epochs_per_phase: int = 3,
    seed: int = 42,
) -> ExperimentSpec:
    run_dir = ROOT / "runs" / f"decoupling_{name}"
    cmd = [
        PYTHON, str(ROOT / "scripts" / "train_hybrid_curriculum.py"),
        "--output-dir", str(run_dir),
        "--use-default-curriculum", "True",
        "--seed", str(seed),
        "--episodes", "15",
        "--history-phase-epochs", str(epochs_per_phase),
        "--history-history-supervision-weight", "0.4",
    ]
    return ExperimentSpec(
        name=name,
        agent="hybrid",
        task="rdm_macaque",
        run_dir=run_dir,
        train_cmd=cmd,
        description=description,
    )


def build_experiments(*, include_hybrid: bool, quick: bool) -> list[ExperimentSpec]:
    """Define the experiment matrix."""
    epochs = 15 if quick else 30

    experiments = [
        _rddm_spec(
            "A_rddm_control",
            "R-DDM: WFPT only, no history (control)",
            wfpt=0.5, history=0.0, per_trial=0.0, epochs=epochs,
        ),
        _rddm_spec(
            "B_rddm_batch_history",
            "R-DDM: WFPT + batch-mean history (old approach)",
            wfpt=0.5, history=0.5, per_trial=0.0, epochs=epochs,
        ),
        _rddm_spec(
            "C_rddm_pertrial",
            "R-DDM: WFPT + per-trial history (KEY TEST)",
            wfpt=0.5, history=0.0, per_trial=0.5, epochs=epochs,
        ),
        _rddm_spec(
            "D_rddm_combined",
            "R-DDM: WFPT + batch + per-trial (combined)",
            wfpt=0.5, history=0.3, per_trial=0.5, epochs=epochs,
        ),
    ]

    if include_hybrid:
        experiments.append(
            _hybrid_spec(
                "E_hybrid_pertrial",
                "Hybrid: default curriculum + per-trial history in final phase",
                per_trial_history=0.5,
                epochs_per_phase=3 if quick else 5,
            ),
        )

    return experiments


def evaluate_run(run_dir: Path) -> dict | None:
    """Run evaluate_agent.py on a completed run and return parsed metrics."""
    trials_path = run_dir / "trials.ndjson"
    if not trials_path.exists():
        # Try best checkpoint rollout
        trials_path = run_dir / "trials_best.ndjson"
    if not trials_path.exists():
        print(f"  [WARN] No trials file found in {run_dir}")
        return None

    cmd = [PYTHON, str(ROOT / "scripts" / "evaluate_agent.py"), "--run", str(run_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [WARN] Evaluation failed: {result.stderr[:200]}")
        return None

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print("  [WARN] No metrics.json produced")
        return None

    with metrics_path.open() as f:
        return json.load(f)


def extract_key_metrics(metrics: dict) -> dict[str, float | str]:
    """Pull the metrics that matter for the Decoupling experiment."""
    # metrics.json has {"log": ..., "metrics": {...}} structure
    inner = metrics.get("metrics", metrics)
    psycho = inner.get("psychometric", {})
    chrono = inner.get("chronometric", {})
    history = inner.get("history", {})
    quality = inner.get("quality", {})

    return {
        "psych_slope": psycho.get("slope"),
        "bias": psycho.get("bias"),
        "chrono_slope": chrono.get("slope_ms_per_unit"),
        "corrected_slope": chrono.get("corrected_slope"),
        "ceiling_frac": chrono.get("ceiling_fraction"),
        "win_stay": history.get("win_stay"),
        "lose_shift": history.get("lose_shift"),
        "sticky": history.get("sticky_choice"),
        "quality": quality.get("overall", "?"),
    }


def print_comparison_table(
    results: list[tuple[ExperimentSpec, dict[str, float | str]]],
) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 120)
    print("DECOUPLING EXPERIMENT — COMPARISON TABLE")
    print("=" * 120)

    # Reference values
    print(f"\n{'Reference':20s} | {'PsychSlope':>10s} | {'Bias':>8s} | {'ChronoSlope':>12s} | "
          f"{'WinStay':>8s} | {'LoseShift':>10s} | {'Sticky':>8s} | {'Quality':>8s}")
    print(f"{'IBL Mouse':20s} | {'13.2':>10s} | {'0.07':>8s} | {'-36':>12s} | "
          f"{'0.73':>8s} | {'0.43':>10s} | {'—':>8s} | {'—':>8s}")
    print(f"{'Macaque RDM':20s} | {'17.6':>10s} | {'~0':>8s} | {'-645':>12s} | "
          f"{'0.49':>8s} | {'0.52':>10s} | {'—':>8s} | {'—':>8s}")
    print("-" * 120)

    for spec, kv in results:
        def _fmt(v: float | str | None, width: int = 8) -> str:
            if v is None:
                return "N/A".rjust(width)
            if isinstance(v, float):
                return f"{v:.3f}".rjust(width)
            return str(v).rjust(width)

        row = (
            f"{spec.name:20s} | "
            f"{_fmt(kv['psych_slope'], 10)} | "
            f"{_fmt(kv['bias'], 8)} | "
            f"{_fmt(kv['chrono_slope'], 12)} | "
            f"{_fmt(kv['win_stay'], 8)} | "
            f"{_fmt(kv['lose_shift'], 10)} | "
            f"{_fmt(kv['sticky'], 8)} | "
            f"{_fmt(kv['quality'], 8)}"
        )
        print(row)

    print("=" * 120)

    # Assess success criteria
    print("\nSUCCESS CRITERIA: win-stay > 0.6 AND chrono_slope < -100 ms/unit in the same run")
    for spec, kv in results:
        ws = kv.get("win_stay")
        cs = kv.get("chrono_slope")
        if ws is not None and cs is not None and isinstance(ws, float) and isinstance(cs, float):
            if ws > 0.6 and cs < -100:
                print(f"  ✓ {spec.name}: DECOUPLING SOLVED (win-stay={ws:.3f}, slope={cs:.1f})")
            elif ws > 0.6:
                print(f"  ~ {spec.name}: History OK but flat chronometric (win-stay={ws:.3f}, slope={cs:.1f})")
            elif cs < -100:
                print(f"  ~ {spec.name}: Chronometric OK but weak history (win-stay={ws:.3f}, slope={cs:.1f})")
            else:
                print(f"  ✗ {spec.name}: Neither criterion met (win-stay={ws:.3f}, slope={cs:.1f})")
        else:
            print(f"  ? {spec.name}: Metrics unavailable")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-hybrid", action="store_true", help="Also run Hybrid experiment E")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs for smoke testing")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training, just evaluate existing runs")
    args = parser.parse_args()

    experiments = build_experiments(include_hybrid=args.include_hybrid, quick=args.quick)

    results: list[tuple[ExperimentSpec, dict[str, float | str]]] = []

    for i, spec in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {spec.name}")
        print(f"  {spec.description}")
        print(f"  Run dir: {spec.run_dir}")
        print(f"{'='*80}")

        if not args.evaluate_only:
            print(f"\n  Training {spec.agent}...")
            result = subprocess.run(
                spec.train_cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("  [ERROR] Training failed!")
                print(f"  stderr: {result.stderr[:500]}")
                continue
            # Print last few lines of stdout for progress visibility
            stdout_lines = result.stdout.strip().split("\n")
            for line in stdout_lines[-5:]:
                print(f"  {line}")

        print("\n  Evaluating...")
        metrics = evaluate_run(spec.run_dir)
        if metrics is not None:
            kv = extract_key_metrics(metrics)
            results.append((spec, kv))
            print(f"  Key metrics: slope={kv['psych_slope']}, chrono={kv['chrono_slope']}, "
                  f"WS={kv['win_stay']}, LS={kv['lose_shift']}")
        else:
            print("  [SKIP] Could not evaluate")

    if results:
        print_comparison_table(results)
    else:
        print("\n[ERROR] No experiments completed successfully.")


if __name__ == "__main__":
    main()
