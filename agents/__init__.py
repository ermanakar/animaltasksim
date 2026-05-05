"""Baseline agents shipped with AnimalTaskSim."""

from agents.adaptive_control_agent import (
    RECOMMENDED_ADAPTIVE_CONTROL_PROFILE,
    AdaptiveControlConfig,
    AdaptiveControlModel,
    AdaptiveControlProfile,
    AdaptiveControlTrainer,
    ResolvedAdaptiveControlProfile,
    adaptive_control_profile_flags,
    train_adaptive_control,
)

__all__ = [
    "AdaptiveControlConfig",
    "AdaptiveControlModel",
    "AdaptiveControlProfile",
    "AdaptiveControlTrainer",
    "RECOMMENDED_ADAPTIVE_CONTROL_PROFILE",
    "ResolvedAdaptiveControlProfile",
    "adaptive_control_profile_flags",
    "train_adaptive_control",
]
