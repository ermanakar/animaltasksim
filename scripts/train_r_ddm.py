#!/usr/bin/env python
"""CLI for training the Recurrent Drift-Diffusion Model (R-DDM) agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from agents.r_ddm import RDDMConfig, RDDMTrainer, RDDMTrainingSchedule


@dataclass(slots=True)
class TrainRDDMArgs:
    """Command-line arguments for training the R-DDM agent."""

    run_dir: Path = Path("runs/r_ddm_experiment")
    """Directory where outputs (model, metrics, rollouts) are stored."""

    task: Literal["ibl_2afc", "rdm_macaque"] = "ibl_2afc"
    reference_log: Path | None = None
    """Path to the reference NDJSON file (inferred from task if omitted)."""

    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 42

    choice_loss_weight: float = 1.0
    wfpt_loss_weight: float = 0.5
    history_loss_weight: float = 0.3
    per_trial_history_weight: float = 0.5
    drift_supervision_weight: float = 0.2

    choice_kl_weight: float = 0.0
    choice_kl_target_slope: float = 6.0
    entropy_loss_weight: float = 0.0
    entropy_target: float = 0.35
    entropy_weight_lr: float = 1e-3
    freeze_bias_epochs: int = 3

    warmup_epochs: int = 6
    choice_warmup_weight: float = 3.0
    enable_wfpt_epoch: int = 6
    wfpt_ramp_epochs: int = 5
    enable_history_epoch: int = 10
    history_ramp_epochs: int = 8

    non_decision_target: float = 0.3
    non_decision_reg_weight: float = 0.1

    prior_feature_scale: float = 0.2
    history_feature_scale: float = 0.5
    stimulus_scale: float = 1.0

    max_sessions: int | None = None
    device: str = "cpu"
    rollout_trials: int = 1200
    motor_delay_ms: float = 200.0
    stochastic_eval: bool = False


def main(args: TrainRDDMArgs) -> None:
    config = RDDMConfig(
        task=args.task,
        reference_log=args.reference_log,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        choice_loss_weight=args.choice_loss_weight,
        wfpt_loss_weight=args.wfpt_loss_weight,
        history=args.history_loss_weight,
        per_trial_history_weight=args.per_trial_history_weight,
        drift_supervision_weight=args.drift_supervision_weight,
        choice_kl_weight=args.choice_kl_weight,
        choice_kl_target_slope=args.choice_kl_target_slope,
        entropy_loss_weight=args.entropy_loss_weight,
        entropy_target=args.entropy_target,
        entropy_weight_lr=args.entropy_weight_lr,
        freeze_bias_epochs=args.freeze_bias_epochs,
        schedule=RDDMTrainingSchedule(
            warmup_epochs=args.warmup_epochs,
            choice_warmup_weight=args.choice_warmup_weight,
            enable_wfpt_epoch=args.enable_wfpt_epoch,
            wfpt_ramp_epochs=args.wfpt_ramp_epochs,
            enable_history_epoch=args.enable_history_epoch,
        ),
        history_ramp_epochs=args.history_ramp_epochs,
        non_decision_target=args.non_decision_target,
        non_decision_reg_weight=args.non_decision_reg_weight,
        prior_feature_scale=args.prior_feature_scale,
        history_feature_scale=args.history_feature_scale,
        stimulus_scale=args.stimulus_scale,
        max_sessions=args.max_sessions,
        device=args.device,
        rollout_trials=args.rollout_trials,
        motor_delay_ms=args.motor_delay_ms,
    )

    trainer = RDDMTrainer(config)
    history = trainer.train()
    rollout_path = trainer.rollout(n_trials=args.rollout_trials, run_name=args.run_dir.name, stochastic=True, balanced=True)

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


if __name__ == "__main__":
    main(tyro.cli(TrainRDDMArgs))
