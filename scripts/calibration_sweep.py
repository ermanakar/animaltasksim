"""Automated hyperparameter sweep for agent calibration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tyro

from agents.hybrid_ddm_lstm import (
    CurriculumConfig,
    HybridTrainingConfig,
    LossWeights,
    train_hybrid,
)


@dataclass
class SweepRunner:
    """Configure and run a calibration sweep."""

    base_curriculum_name: str = "rt_weighted_calibration_curriculum"
    sweep_param_name: str = "rt_soft"
    sweep_values: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2)
    sweep_phase_name: str = "phase6_rt_weighted_calibration"
    base_output_dir: Path = Path("runs/calibration_sweep")
    seed: int = 4321
    num_episodes: int = 10

    def run(self) -> None:
        """Execute the hyperparameter sweep."""
        print(f"Starting calibration sweep for '{self.sweep_param_name}'...")
        print(f"  Values: {self.sweep_values}")
        print(f"  Base curriculum: {self.base_curriculum_name}")
        print(f"  Output directory: {self.base_output_dir}")

        results: list[dict[str, Any]] = []

        for value in self.sweep_values:
            run_name = f"{self.sweep_param_name}_{value:.3f}"
            run_output_dir = self.base_output_dir / run_name
            print(f"\n{'='*80}")
            print(f"Running sweep value: {self.sweep_param_name} = {value}")
            print(f"Output: {run_output_dir}")
            print(f"{'='*80}")

            # 1. Create the custom curriculum
            try:
                base_curriculum_fn = getattr(CurriculumConfig, self.base_curriculum_name)
                curriculum = base_curriculum_fn()
            except AttributeError:
                print(f"Error: Curriculum '{self.base_curriculum_name}' not found.")
                continue

            config_kwargs: dict[str, Any] = {
                "output_dir": run_output_dir,
                "curriculum": curriculum,
                "seed": self.seed,
                "episodes": self.num_episodes,
            }

            # Find and modify the target phase OR the main config
            if self.sweep_param_name in asdict(LossWeights()):
                target_phase_found = False
                for phase in curriculum.phases:
                    if phase.name == self.sweep_phase_name:
                        if hasattr(phase.loss_weights, self.sweep_param_name):
                            setattr(phase.loss_weights, self.sweep_param_name, value)
                            target_phase_found = True
                            break
                if not target_phase_found:
                    print(f"Error: Phase '{self.sweep_phase_name}' not found for loss weight sweep.")
                    continue
            elif self.sweep_param_name in asdict(HybridTrainingConfig()):
                config_kwargs[self.sweep_param_name] = value
            else:
                print(f"Error: sweep_param_name '{self.sweep_param_name}' not recognized.")
                continue

            # 2. Configure and run training
            config = HybridTrainingConfig(**config_kwargs)
            train_hybrid(config)

            # 3. Evaluate the agent
            eval_cmd = [
                sys.executable,
                "scripts/evaluate_agent.py",
                "--run",
                str(run_output_dir),
            ]
            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
            eval_metrics = json.loads(eval_result.stdout)
            
            # 4. Generate the dashboard
            dashboard_cmd = [
                sys.executable,
                "scripts/make_dashboard.py",
                "--opts.agent-log",
                str(run_output_dir / "trials.ndjson"),
                "--opts.reference-log",
                "data/macaque/reference.ndjson",
                "--opts.output",
                str(run_output_dir / "dashboard.html"),
            ]
            subprocess.run(dashboard_cmd, check=True)

            # 5. Store results
            chronometric_metrics = eval_metrics.get("chronometric", {})
            slope = chronometric_metrics.get("slope_ms_per_unit") if chronometric_metrics else None
            results.append(
                {
                    "value": value,
                    "run_name": run_name,
                    "output_dir": str(run_output_dir),
                    "chronometric_slope": slope,
                }
            )

        # 6. Print summary
        print(f"\n{'='*80}")
        print("Calibration Sweep Summary")
        print(f"{'='*80}")
        print(f"{'Value':<10} | {'Chronometric Slope':<20} | {'Run Name'}")
        print(f"{'-'*10} | {'-'*20} | {'-'*30}")
        for result in sorted(results, key=lambda x: x.get("chronometric_slope", 0) or 0):
            slope = result.get('chronometric_slope')
            slope_str = f"{slope:.2f}" if slope is not None else "N/A"
            print(f"{result['value']:<10.3f} | {slope_str:<20} | {result['run_name']}")
        
        summary_path = self.base_output_dir / "sweep_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        print(f"\n✓ Summary saved to {summary_path}")


def main() -> None:
    """CLI entry point."""
    if tyro is None:
        raise ImportError("tyro is required for this script. `pip install tyro`")
    tyro.cli(SweepRunner).run()


if __name__ == "__main__":
    main()
