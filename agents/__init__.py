"""Baseline agents shipped with AnimalTaskSim."""

from agents.adaptive_control_agent import (
    AdaptiveControlConfig,
    AdaptiveControlModel,
    AdaptiveControlTrainer,
    train_adaptive_control,
)

__all__ = [
    "AdaptiveControlConfig",
    "AdaptiveControlModel",
    "AdaptiveControlTrainer",
    "train_adaptive_control",
]
