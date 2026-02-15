#!/usr/bin/env python
"""CLI for training the R-DDM agent with only the choice loss."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import subprocess

import tyro

from agents.r_ddm import RDDMConfig, RDDMTrainer, RDDMTrainingSchedule


@dataclass(slots=True)
class TrainChoiceOnlyArgs:
    """Command-line arguments for training the R-DDM agent with only the choice loss."""

    run_dir: Path = Path("runs/r_ddm_choice_only")
    """Directory where outputs (model, metrics, rollouts) are stored."""

    task: Literal["ibl_2afc", "rdm_macaque"] = "rdm_macaque"
    reference_log: Path | None = None
    """Path to the reference NDJSON file (inferred from task if omitted)."""

    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 42

    entropy_loss_weight: float = 0.1
    freeze_bias_epochs: int = 3

    max_sessions: int | None = None
    device: str = "cpu"
    rollout_trials: int = 1200
    stochastic_eval: bool = False


def main(args: TrainChoiceOnlyArgs) -> None:
    config = RDDMConfig(
        task=args.task,
        reference_log=args.reference_log,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        choice_loss_weight=1.0,
        wfpt_loss_weight=0.0,
        history_loss_weight=0.0,
        drift_supervision_weight=0.0,
        choice_kl_weight=0.0,
        choice_kl_target_slope=6.0,
        entropy_loss_weight=args.entropy_loss_weight,
        schedule=RDDMTrainingSchedule(),
        history_ramp_epochs=0,
        non_decision_target=0.0,
        non_decision_reg_weight=0.0,
        prior_feature_scale=0.0,
        history_feature_scale=0.0,
        stimulus_scale=1.0,
        max_sessions=args.max_sessions,
        device=args.device,
        rollout_trials=args.rollout_trials,
        motor_delay_ms=0.0,
        freeze_bias_epochs=args.freeze_bias_epochs,
    )

    trainer = RDDMTrainer(config)
    history = trainer.train()
    rollout_path = trainer.rollout(
        n_trials=args.rollout_trials, run_name=args.run_dir.name, stochastic=True, balanced=True
    )

    best_checkpoint = args.run_dir / "model_best.pt"
    best_rollout_path: Path | None = None
    if best_checkpoint.exists():
        best_rollout_path = trainer.rollout(
            n_trials=args.rollout_trials,
            seed=args.seed,
            run_name=f"{args.run_dir.name}_best",
            checkpoint_path=best_checkpoint,
            suffix="_best",
            stochastic=True,
            balanced=True,
        )

    print("\nTraining summary:")
    for state in history:
        print(
            f"Epoch {state.epoch:02d} - loss={state.total_loss:.3f} "
            f"choice={state.choice_loss:.3f} wfpt={state.wfpt_loss:.3f} "
            f"history={state.history_loss:.3f} drift={state.drift_loss:.3f} nd={state.non_decision_loss:.3f} "
            f"entropy={state.entropy_loss:.3f} kl={state.choice_kl_loss:.3f} "
            f"acc={state.accuracy:.3f} "
            f"win-stay={state.win_stay_pred:.3f} lose-shift={state.lose_shift_pred:.3f}"
        )
    print(f"\nRollout saved to: {rollout_path}")
    if best_rollout_path is not None:
        print(f"Best-checkpoint rollout saved to: {best_rollout_path}")

    # Evaluate the agent
    subprocess.run(["python3", "scripts/evaluate_agent.py", "--run", str(args.run_dir), "--is-choice-only"], check=True)


if __name__ == "__main__":
    main(tyro.cli(TrainChoiceOnlyArgs))
