"""Curriculum configurations for the hybrid DDM + LSTM agent."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from agents.losses import LossWeights

@dataclass(slots=True)
class CurriculumPhase:
    """Configuration for a single curriculum learning phase."""

    name: str
    epochs: int
    loss_weights: LossWeights
    success_criteria: dict[str, float] = field(default_factory=dict)
    min_commit_steps: Optional[int] = None
    max_commit_steps: Optional[int] = None
    freeze_except_history_bias: bool = False  # Only train history_bias_head
    history_bias_lr: Optional[float] = None  # Dedicated LR for history_bias_head
    # Success criteria keys: 'min_slope_abs', 'min_r2', 'min_rt_diff_abs'


@dataclass(slots=True)
class CurriculumConfig:
    """Multi-phase curriculum learning schedule."""

    phases: List[CurriculumPhase] = field(default_factory=list)
    allow_early_stopping: bool = True
    checkpoint_each_phase: bool = True

    @staticmethod
    def default_rt_first() -> CurriculumConfig:
        """Default 2-phase curriculum: WFPT → full balance."""
        phase1 = CurriculumPhase(
            name="phase1_wfpt_only",
            epochs=15,
            loss_weights=LossWeights(
                choice=0.0,
                rt=0.0,
                history=0.0,
                drift_supervision=0.25,
                non_decision_supervision=0.15,
                wfpt=1.0,
            ),
            success_criteria={"min_slope_abs": 250.0, "min_r2": 0.1, "max_slope": -150.0},
        )
        phase2 = CurriculumPhase(
            name="phase2_full_balance",
            epochs=5,
            loss_weights=LossWeights(
                choice=1.0,
                rt=0.0,
                history=0.1,
                drift_supervision=0.05,
                non_decision_supervision=0.05,
                wfpt=0.5,
                twin_supervision=0.1,
            ),
            success_criteria={},
        )
        return CurriculumConfig(phases=[phase1, phase2])

    @staticmethod
    def wfpt_history_refine() -> CurriculumConfig:
        """Extended curriculum that adds a history-aware refinement phase."""
        base = CurriculumConfig.default_rt_first()
        phase3 = CurriculumPhase(
            name="phase3_history_tune",
            epochs=5,
            loss_weights=LossWeights(
                choice=0.8,
                rt=0.1,
                history=0.25,
                drift_supervision=0.02,
                non_decision_supervision=0.1,
                wfpt=0.3,
            ),
            success_criteria={},
        )
        phases = list(base.phases) + [phase3]
        return CurriculumConfig(phases=phases, allow_early_stopping=base.allow_early_stopping, checkpoint_each_phase=base.checkpoint_each_phase)

    @staticmethod
    def wfpt_time_cost() -> CurriculumConfig:
        """Curriculum that enforces WFPT dynamics then adds RT pressure with a reduced commit window."""
        phase1, phase2 = CurriculumConfig.default_rt_first().phases

        phase2_adj = CurriculumPhase(
            name="phase2_balanced_rt_seed",
            epochs=6,
            loss_weights=LossWeights(
                choice=1.0,
                rt=0.0,
                rt_soft=0.1,
                history=0.12,
                drift_supervision=0.05,
                non_decision_supervision=0.1,
                wfpt=0.9,
            ),
            max_commit_steps=300,
        )

        phase3 = CurriculumPhase(
            name="phase3_time_cost",
            epochs=6,
            loss_weights=LossWeights(
                choice=0.9,
                rt=0.0,
                rt_soft=0.1,
                history=0.12,
                drift_supervision=0.02,
                non_decision_supervision=0.1,
                wfpt=0.7,
            ),
            max_commit_steps=300,
        )

        phases = [phase1, phase2_adj, phase3]
        return CurriculumConfig(phases=phases, allow_early_stopping=True)

    @staticmethod
    def guarded_wfpt_dominant() -> CurriculumConfig:
        """A 3-phase curriculum with strong WFPT dominance, adaptive commit windows, and slope/history guardrails."""
        phase1 = CurriculumPhase(
            name="phase1_wfpt_warmup",
            epochs=15,
            loss_weights=LossWeights(
                choice=0.0,
                rt=0.0,
                wfpt=1.0,
                drift_supervision=0.25,
                non_decision_supervision=0.15,
            ),
            success_criteria={
                "min_slope": -800.0,
                "max_slope": -200.0,
                "max_sticky_choice": 0.8,
            },
        )
        phase2 = CurriculumPhase(
            name="phase2_choice_and_relax_window",
            epochs=8,
            loss_weights=LossWeights(
                choice=1.0,
                rt=0.0,
                wfpt=0.95,
                history=0.05,
                drift_supervision=0.05,
                non_decision_supervision=0.05,
            ),
            max_commit_steps=300,
            success_criteria={
                "min_slope": -800.0,
                "max_slope": -200.0,
                "max_sticky_choice": 0.8,
            },
        )
        phase3 = CurriculumPhase(
            name="phase3_finetune",
            epochs=8,
            loss_weights=LossWeights(
                choice=1.0,
                rt=0.0,
                wfpt=0.95,
                history=0.05,
                rt_soft=0.05,
                non_decision_supervision=0.05,
            ),
            max_commit_steps=300,
            success_criteria={},  # Final phase
        )
        return CurriculumConfig(phases=[phase1, phase2, phase3])

    @staticmethod
    def focused_choice_curriculum() -> CurriculumConfig:
        """A curriculum that heavily penalizes incorrect choices to force stimulus attention."""
        phase1 = CurriculumPhase(
            name="phase1_wfpt_warmup",
            epochs=15,
            loss_weights=LossWeights(
                choice=0.0,
                rt=0.0,
                wfpt=1.0,
                drift_supervision=0.2,
                non_decision_supervision=0.15,
            ),
            success_criteria={
                "min_slope": -800.0,
                "max_slope": -200.0,
                "max_sticky_choice": 0.8,
            },
        )
        phase2 = CurriculumPhase(
            name="phase2_strong_choice_penalty",
            epochs=8,
            loss_weights=LossWeights(
                choice=5.0,  # Heavily weight choice loss
                rt=0.0,
                wfpt=0.95,
                history=0.1,
                drift_supervision=0.1,
                non_decision_supervision=0.05,
            ),
            max_commit_steps=300,
            success_criteria={
                "min_slope": -800.0,
                "max_slope": -200.0,
                "max_sticky_choice": 0.8,
            },
        )
        phase3 = CurriculumPhase(
            name="phase3_finetune_with_choice",
            epochs=8,
            loss_weights=LossWeights(
                choice=5.0,
                rt=0.0,
                wfpt=0.95,
                history=0.1,
                rt_soft=0.05,
                drift_supervision=0.1,
                non_decision_supervision=0.05,
            ),
            max_commit_steps=300,
            success_criteria={},
        )
        return CurriculumConfig(phases=[phase1, phase2, phase3])

    @staticmethod
    def annealed_choice_curriculum() -> CurriculumConfig:
        """A curriculum that gradually introduces the choice penalty to preserve the chronometric slope."""
        phase1 = CurriculumPhase(
            name="phase1_wfpt_warmup",
            epochs=15,
            loss_weights=LossWeights(
                choice=0.0,
                rt=0.0,
                wfpt=1.0,
                drift_supervision=0.2,
                non_decision_supervision=0.15,
            ),
            success_criteria={
                "min_slope": -800.0,
                "max_slope": -200.0,
                "max_sticky_choice": 0.8,
            },
        )
        phase2 = CurriculumPhase(
            name="phase2_gentle_choice_intro",
            epochs=8,
            loss_weights=LossWeights(
                choice=1.5,  # Gentle introduction of choice loss
                rt=0.0,
                wfpt=0.95,
                history=0.1,
                drift_supervision=0.15,  # Keep drift supervision high
                non_decision_supervision=0.05,
            ),
            max_commit_steps=300,
            success_criteria={
                "min_slope": -800.0,
                "max_slope": -200.0,
                "max_sticky_choice": 0.8,
            },
        )
        phase3 = CurriculumPhase(
            name="phase3_finetune_annealed",
            epochs=8,
            loss_weights=LossWeights(
                choice=2.5,  # Increase choice weight
                rt=0.0,
                wfpt=0.95,
                history=0.1,
                rt_soft=0.05,
                drift_supervision=0.1,
                non_decision_supervision=0.05,
            ),
            max_commit_steps=300,
            success_criteria={},
        )
        return CurriculumConfig(phases=[phase1, phase2, phase3])

    @staticmethod
    def history_supervision_curriculum() -> CurriculumConfig:
        """A 4-phase curriculum that adds a final history supervision phase."""
        base = CurriculumConfig.annealed_choice_curriculum()
        phase4 = CurriculumPhase(
            name="phase4_history_supervision",
            epochs=5,
            loss_weights=LossWeights(
                choice=2.5,
                rt=0.0,
                wfpt=0.9,
                history=0.0,
                rt_soft=0.05,
                drift_supervision=0.1,
                non_decision_supervision=0.05,
                history_supervision=0.2,  # Activate history supervision
            ),
            max_commit_steps=300,
            success_criteria={},
        )
        return CurriculumConfig(phases=base.phases + [phase4])

    @staticmethod
    def rt_calibration_curriculum() -> CurriculumConfig:
        """A curriculum that adds a final RT calibration phase with scheduled weights."""
        base = CurriculumConfig.history_supervision_curriculum()
        phase5 = CurriculumPhase(
            name="phase5_rt_calibration",
            epochs=5,
            loss_weights=LossWeights(
                choice=2.5,
                rt=0.0,
                wfpt=0.9,
                history=0.0,
                rt_soft=0.1,  # Activate soft RT penalty
                drift_supervision=0.1,
                non_decision_supervision=0.05,
                history_supervision=0.2,
            ),
            max_commit_steps=300,
            success_criteria={},
        )
        return CurriculumConfig(phases=base.phases + [phase5])

    @staticmethod
    def rt_weighted_calibration_curriculum() -> CurriculumConfig:
        """A curriculum that adds a final RT calibration phase with scheduled weights."""
        base = CurriculumConfig.rt_calibration_curriculum()
        phase6 = CurriculumPhase(
            name="phase6_rt_weighted_calibration",
            epochs=5,
            loss_weights=LossWeights(
                choice=2.5,
                rt=0.0,
                wfpt=0.9,
                history=0.0,
                rt_soft=0.1,  # Activate soft RT penalty
                drift_supervision=0.1,
                non_decision_supervision=0.05,
                history_supervision=0.2,
            ),
            max_commit_steps=300,
            success_criteria={},
        )
        return CurriculumConfig(phases=base.phases + [phase6])

    @staticmethod
    def drift_scale_calibration_curriculum() -> CurriculumConfig:
        """A curriculum designed to be more stable for higher drift_scale values."""
        base = CurriculumConfig.rt_weighted_calibration_curriculum()
        
        # Modify phase1 to allow for a steeper slope
        phase1_modified = base.phases[0]
        phase1_modified.success_criteria["min_slope"] = -1500.0
        
        # Modify phase2 to be more gentle, protecting the slope
        phase2_modified = CurriculumPhase(
            name="phase2_very_gentle_choice_intro",
            epochs=8,
            loss_weights=LossWeights(
                choice=0.75,  # Reduced choice weight
                rt=0.0,
                wfpt=1.0,  # Increased WFPT weight to maintain RT structure
                history=0.1,
                drift_supervision=0.2,  # Increased drift supervision
                non_decision_supervision=0.05,
            ),
            max_commit_steps=300,
            success_criteria={
                "min_slope": -1500.0,
                "max_slope": -200.0,
                "max_sticky_choice": 0.8,
            },
        )
        base.phases[0] = phase1_modified
        base.phases[1] = phase2_modified
        return base

    @staticmethod
    def history_finetune_curriculum(
        *,
        history_phase_epochs: int = 5,
        history_choice_weight: float = 2.5,
        history_wfpt_weight: float = 0.9,
        history_history_weight: float = 0.0,
        history_rt_soft_weight: float = 0.1,
        history_drift_supervision_weight: float = 0.1,
        history_non_decision_supervision_weight: float = 0.05,
        history_history_supervision_weight: float = 0.4,
        history_per_trial_history_weight: float = 0.5,
        history_max_commit_steps: int = 120,
        history_freeze_except_history_bias: bool = True,
        history_bias_lr: float = 3e-3,
    ) -> CurriculumConfig:
        """A curriculum that adds a final history finetuning phase."""
        base = CurriculumConfig.drift_scale_calibration_curriculum()
        phase7 = CurriculumPhase(
            name="phase7_history_finetune",
            epochs=history_phase_epochs,
            loss_weights=LossWeights(
                choice=history_choice_weight,
                rt=0.0,
                wfpt=history_wfpt_weight,
                history=history_history_weight,
                rt_soft=history_rt_soft_weight,
                drift_supervision=history_drift_supervision_weight,
                non_decision_supervision=history_non_decision_supervision_weight,
                history_supervision=history_history_supervision_weight,
                per_trial_history=history_per_trial_history_weight,
            ),
            max_commit_steps=history_max_commit_steps,
            success_criteria={},
            freeze_except_history_bias=history_freeze_except_history_bias,
            history_bias_lr=history_bias_lr,
        )
        return CurriculumConfig(phases=base.phases + [phase7])
