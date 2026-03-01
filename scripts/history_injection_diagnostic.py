#!/usr/bin/env python
"""History injection diagnostic.

Tests whether the DDM mechanism can express the right win-stay/lose-shift
behavior when given the correct stay_tendency inputs, bypassing the
history networks entirely.

Loads a pretrained model (good psych/chrono), overrides stay_tendency
with fixed values based on win/lose outcome, runs rollout-only (no
training), evaluates metrics. Fast: ~2-3 min per combo.

This answers the critical question: is the problem that the networks
can't LEARN the right signal, or that the DDM can't EXPRESS the right
behavior even with perfect inputs?
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

from agents.hybrid_config import HybridTrainingConfig
from agents.hybrid_trainer import HybridDDMTrainer


@dataclass(slots=True)
class InjectionArgs:
    """History injection diagnostic arguments."""

    run_root: Path = Path("runs/history_injection_diagnostic")
    """Root directory where diagnostic runs are stored."""

    pretrained_run: Path = Path("runs/drift_calibration_v2/target6p0_seed42")
    """Pretrained model to load (must have model.pt and config.json)."""

    win_tendencies: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    )
    """Fixed stay_tendency values to inject after wins (positive = stay)."""

    lose_tendencies: list[float] = field(
        default_factory=lambda: [-0.1, 0.0, 0.1, 0.2, 0.3]
    )
    """Fixed stay_tendency values to inject after losses."""

    history_bias_scale: float = 2.0
    """Override model's history_bias_scale (pretrained model may have old value)."""

    history_drift_scale: float = 0.3
    """Override model's history_drift_scale (pretrained model may have 0.0)."""

    episodes: int = 20
    """Rollout episodes per combo."""

    seed: int = 42
    """Random seed for rollout."""

    dry_run: bool = False
    """Print combos without executing."""


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
                "degenerate": data.get("quality", {}).get("degenerate"),
            }
        )
    return summary


def main(args: InjectionArgs) -> None:
    """Run the history injection diagnostic."""
    args.run_root.mkdir(parents=True, exist_ok=True)

    # Verify pretrained model exists
    model_path = args.pretrained_run / "model.pt"
    if not model_path.exists():
        print(f"[ERROR] Pretrained model not found: {model_path}")
        sys.exit(1)

    combos = list(
        itertools.product(args.win_tendencies, args.lose_tendencies)
    )
    summaries: list[dict[str, object]] = []
    failures: list[str] = []

    print(f"\n{'='*70}")
    print("HISTORY INJECTION DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"Pretrained model: {args.pretrained_run}")
    print(f"win_tendencies: {args.win_tendencies}")
    print(f"lose_tendencies: {args.lose_tendencies}")
    print(f"Total combos: {len(combos)}")
    print(f"history_bias_scale override: {args.history_bias_scale}")
    print(f"history_drift_scale override: {args.history_drift_scale}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.run_root}")
    print(f"{'='*70}\n")

    if args.dry_run:
        for i, (win_t, lose_t) in enumerate(combos, 1):
            run_name = f"win{win_t}_lose{lose_t}".replace(".", "p").replace("-", "n")
            print(f"[{i}/{len(combos)}] {run_name}")
        print(f"\n{len(combos)} combos would be run.")
        return

    # Load pretrained weights once
    pretrained_state = torch.load(model_path, map_location="cpu", weights_only=True)
    print(f"[INFO] Loaded pretrained model from {model_path}")

    for i, (win_t, lose_t) in enumerate(combos, 1):
        run_name = f"win{win_t}_lose{lose_t}".replace(".", "p").replace("-", "n")
        run_dir = args.run_root / run_name

        print(f"\n{'='*70}")
        print(f"[{i}/{len(combos)}] {run_name}")
        print(f"  inject_win_tendency={win_t}, inject_lose_tendency={lose_t}")
        print(f"{'='*70}")

        try:
            # Create trainer with injection config
            config = HybridTrainingConfig(
                task="ibl_2afc",
                output_dir=run_dir,
                seed=args.seed,
                episodes=args.episodes,
                trials_per_episode=400,
                hidden_size=64,
                drift_scale=10.0,
                drift_magnitude_target=6.0,
                max_commit_steps=300,
                min_commit_steps=5,
                step_ms=10,
                lapse_rate=0.05,
                history_bias_scale=args.history_bias_scale,
                history_drift_scale=args.history_drift_scale,
                inject_win_tendency=win_t,
                inject_lose_tendency=lose_t,
            )

            trainer = HybridDDMTrainer(config)

            # Load pretrained weights
            trainer.model.load_state_dict(pretrained_state)

            # Override scale params to desired values (pretrained model may have old values)
            trainer.model.history_bias_scale.data.fill_(args.history_bias_scale)
            trainer.model.history_drift_scale.data.fill_(args.history_drift_scale)

            trainer.model.eval()

            # Run rollout only (no training)
            paths = config.output_paths()
            trainer.rollout(paths)

            # Save config for evaluate_agent.py
            config_payload = {
                "task": config.task,
                "inject_win_tendency": win_t,
                "inject_lose_tendency": lose_t,
                "history_bias_scale": args.history_bias_scale,
                "history_drift_scale": args.history_drift_scale,
                "pretrained_run": str(args.pretrained_run),
                "episodes": args.episodes,
                "seed": args.seed,
            }
            (run_dir / "config.json").write_text(
                json.dumps(config_payload, indent=2), encoding="utf-8"
            )

        except Exception as e:
            print(f"[WARN] Rollout failed for {run_name}: {e}")
            failures.append(run_name)
            continue

        # Evaluate
        cmd_eval = [
            sys.executable,
            "scripts/evaluate_agent.py",
            "--run",
            str(run_dir),
        ]
        print("[CMD]", " ".join(cmd_eval))
        rc = subprocess.run(cmd_eval, check=False).returncode
        if rc != 0:
            print(f"[WARN] Evaluation failed for {run_name}")
            failures.append(run_name)
            continue

        row: dict[str, object] = {
            "run": run_name,
            "win_tendency": win_t,
            "lose_tendency": lose_t,
        }
        row.update(summarise_run(run_dir))
        summaries.append(row)

        # Print key metrics inline
        ws = row.get("win_stay")
        ls = row.get("lose_shift")
        ps = row.get("psych_slope")
        cs = row.get("chrono_slope")
        print(
            f"\n  >> win_stay={ws}, lose_shift={ls}, "
            f"psych={ps}, chrono={cs}"
        )

    # Write CSV summary
    if summaries:
        csv_path = args.run_root / "sweep_summary.csv"
        fieldnames = list(summaries[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        print(f"\n[INFO] Summary saved to {csv_path}")

        # Print results table
        print(f"\n{'='*70}")
        print("RESULTS: win_tendency x lose_tendency -> win_stay / lose_shift")
        print(f"{'='*70}")
        print(
            f"{'win_t':>8} | {'lose_t':>8} | {'win_stay':>10} | "
            f"{'lose_shift':>10} | {'psych_slope':>12} | {'chrono_slope':>12}"
        )
        print("-" * 75)

        for win_t in args.win_tendencies:
            for lose_t in args.lose_tendencies:
                rows = [
                    r
                    for r in summaries
                    if r["win_tendency"] == win_t
                    and r["lose_tendency"] == lose_t
                ]
                if not rows:
                    continue
                r = rows[0]

                def _val(v: object) -> str:
                    if v is None:
                        return "N/A"
                    return f"{v:.3f}" if isinstance(v, float) else str(v)

                print(
                    f"{win_t:>8.1f} | {lose_t:>8.1f} | "
                    f"{_val(r.get('win_stay')):>10} | "
                    f"{_val(r.get('lose_shift')):>10} | "
                    f"{_val(r.get('psych_slope')):>12} | "
                    f"{_val(r.get('chrono_slope')):>12}"
                )
            print()  # Blank line between win_tendency groups

        print(
            "Targets: win_stay ~ 0.724, lose_shift ~ 0.427, "
            "psych_slope ~ 13.2, chrono_slope ~ -36 ms/unit"
        )

    if failures:
        print(f"\n[WARN] {len(failures)} combos failed: {failures}")


if __name__ == "__main__":
    main(tyro.cli(InjectionArgs))
