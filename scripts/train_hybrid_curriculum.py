"""Train the hybrid DDM + LSTM agent using curriculum learning."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import tyro  # type: ignore
except ImportError:  # pragma: no cover - tyro optional
    tyro = None

from agents.hybrid_ddm_lstm import (
    CurriculumConfig,
    CurriculumPhase,
    HybridTrainingConfig,
    LossWeights,
    train_hybrid,
)


@dataclass(slots=True)
class LossWeightArgs:
    choice: float = 1.0
    rt: float = 1.0
    history: float = 0.0
    drift_supervision: float = 0.0


@dataclass(slots=True)
class TrainCurriculumArgs:
    """Arguments for curriculum learning training."""
    
    reference_log: Path = Path("data/macaque/reference.ndjson")
    output_dir: Path = Path("runs/rdm_hybrid_curriculum")
    agent_version: str = "0.1.0"
    trials_per_episode: int = 400
    episodes: int = 20
    seed: int = 42
    hidden_size: int = 64
    learning_rate: float = 3e-4
    step_ms: int = 10
    max_sessions: int | None = None
    max_trials_per_session: int | None = None
    min_commit_steps: int = 5
    max_commit_steps: int = 300
    drift_scale: float = 14.0
    
    # Curriculum configuration
    use_default_curriculum: bool = True
    allow_early_stopping: bool = True
    checkpoint_each_phase: bool = True
    
    # Phase 1: RT structure only
    phase1_epochs: int = 5
    phase1_choice_weight: float = 0.0
    phase1_rt_weight: float = 1.0
    phase1_history_weight: float = 0.0
    phase1_drift_supervision_weight: float = 0.5
    phase1_min_slope: float = 100.0  # Minimum |slope| in ms/unit
    phase1_max_slope: float | None = None
    phase1_min_r2: float = 0.1
    phase1_min_rt_diff: float = 50.0  # Minimum RT difference in ms
    
    # Phase 2: Add choice gradually
    phase2_epochs: int = 5
    phase2_choice_weight: float = 0.3
    phase2_rt_weight: float = 0.8
    phase2_history_weight: float = 0.05
    phase2_drift_supervision_weight: float = 0.3
    phase2_min_slope: float = 80.0
    phase2_max_slope: float | None = None
    phase2_min_r2: float = 0.08
    
    # Phase 3: Full balance
    phase3_epochs: int = 5
    phase3_choice_weight: float = 1.0
    phase3_rt_weight: float = 0.5
    phase3_history_weight: float = 0.1
    phase3_drift_supervision_weight: float = 0.1

    # History finetune overrides (used when default curriculum is selected)
    history_phase_epochs: int = 5
    history_choice_weight: float = 2.5
    history_wfpt_weight: float = 0.9
    history_history_weight: float = 0.0
    history_rt_soft_weight: float = 0.1
    history_drift_supervision_weight: float = 0.1
    history_non_decision_supervision_weight: float = 0.05
    history_history_supervision_weight: float = 0.4
    history_per_trial_history_weight: float = 0.0
    history_max_commit_steps: int = 300


def main(args: TrainCurriculumArgs) -> None:
    """Execute curriculum training."""
    
    # Build curriculum config
    if args.use_default_curriculum:
        curriculum = CurriculumConfig.history_finetune_curriculum(
            history_phase_epochs=args.history_phase_epochs,
            history_choice_weight=args.history_choice_weight,
            history_wfpt_weight=args.history_wfpt_weight,
            history_history_weight=args.history_history_weight,
            history_rt_soft_weight=args.history_rt_soft_weight,
            history_drift_supervision_weight=args.history_drift_supervision_weight,
            history_non_decision_supervision_weight=args.history_non_decision_supervision_weight,
            history_history_supervision_weight=args.history_history_supervision_weight,
            history_per_trial_history_weight=args.history_per_trial_history_weight,
            history_max_commit_steps=args.history_max_commit_steps,
        )
        curriculum.allow_early_stopping = args.allow_early_stopping
        curriculum.checkpoint_each_phase = args.checkpoint_each_phase
    else:
        # Custom curriculum from args
        phase1 = CurriculumPhase(
            name="phase1_rt_only",
            epochs=args.phase1_epochs,
            loss_weights=LossWeights(
                choice=args.phase1_choice_weight,
                rt=args.phase1_rt_weight,
                history=args.phase1_history_weight,
                drift_supervision=args.phase1_drift_supervision_weight,
            ),
            success_criteria={
                "min_slope_abs": args.phase1_min_slope,
                **({"max_slope_abs": args.phase1_max_slope} if args.phase1_max_slope is not None else {}),
                "min_r2": args.phase1_min_r2,
                "min_rt_diff_abs": args.phase1_min_rt_diff,
            },
        )
        phase2 = CurriculumPhase(
            name="phase2_add_choice",
            epochs=args.phase2_epochs,
            loss_weights=LossWeights(
                choice=args.phase2_choice_weight,
                rt=args.phase2_rt_weight,
                history=args.phase2_history_weight,
                drift_supervision=args.phase2_drift_supervision_weight,
            ),
            success_criteria={
                "min_slope_abs": args.phase2_min_slope,
                **({"max_slope_abs": args.phase2_max_slope} if args.phase2_max_slope is not None else {}),
                "min_r2": args.phase2_min_r2,
            },
        )
        phase3 = CurriculumPhase(
            name="phase3_full_balance",
            epochs=args.phase3_epochs,
            loss_weights=LossWeights(
                choice=args.phase3_choice_weight,
                rt=args.phase3_rt_weight,
                history=args.phase3_history_weight,
                drift_supervision=args.phase3_drift_supervision_weight,
            ),
            success_criteria={},  # Final phase, no hard criteria
        )
        curriculum = CurriculumConfig(
            phases=[phase1, phase2, phase3],
            allow_early_stopping=args.allow_early_stopping,
            checkpoint_each_phase=args.checkpoint_each_phase,
        )
    
    # Build training config
    config = HybridTrainingConfig(
        reference_log=args.reference_log,
        output_dir=args.output_dir,
        agent_version=args.agent_version,
        trials_per_episode=args.trials_per_episode,
        episodes=args.episodes,
        seed=args.seed,
        epochs=5,  # Will be overridden by curriculum phases
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        loss_weights=LossWeights(),  # Will be overridden by curriculum
        step_ms=args.step_ms,
        max_sessions=args.max_sessions,
        max_trials_per_session=args.max_trials_per_session,
        min_commit_steps=args.min_commit_steps,
        max_commit_steps=args.max_commit_steps,
        drift_scale=args.drift_scale,
        curriculum=curriculum,
    )
    
    print(f"\n{'='*80}")
    print("CURRICULUM LEARNING TRAINING")
    print(f"{'='*80}")
    print(f"Output: {config.output_dir}")
    print(f"Phases: {len(curriculum.phases)}")
    print(f"Early stopping: {curriculum.allow_early_stopping}")
    print(f"Checkpoints: {curriculum.checkpoint_each_phase}")
    print(f"Seed: {config.seed}")
    print(f"{'='*80}\n")
    
    result = train_hybrid(config)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    if "phases" in result:
        print(f"\nPhase Summary:")
        for phase_result in result["phases"]:
            status = "✓ PASSED" if phase_result["success"] else "✗ FAILED"
            print(f"  {phase_result['name']}: {status}")
            if phase_result.get("metrics"):
                for key, val in phase_result["metrics"].items():
                    print(f"    {key}: {val:.4f}")
    print(f"\nFinal artifacts:")
    for key, path in result["paths"].items():
        print(f"  {key}: {path}")
    print(f"\n{'='*80}\n")


def _parse_args() -> TrainCurriculumArgs:
    if tyro is not None:
        return tyro.cli(TrainCurriculumArgs)
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_log", type=Path, default=Path("data/macaque/reference.ndjson"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/rdm_hybrid_curriculum"))
    parser.add_argument("--agent_version", type=str, default="0.1.0")
    parser.add_argument("--trials_per_episode", type=int, default=400)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--drift_scale", type=float, default=10.0)
    parser.add_argument("--use_default_curriculum", type=bool, default=True)
    
    args = parser.parse_args()
    return TrainCurriculumArgs(
        reference_log=args.reference_log,
        output_dir=args.output_dir,
        agent_version=args.agent_version,
        trials_per_episode=args.trials_per_episode,
        episodes=args.episodes,
        seed=args.seed,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        drift_scale=args.drift_scale,
        use_default_curriculum=args.use_default_curriculum,
    )


if __name__ == "__main__":
    args = _parse_args()
    main(args)
