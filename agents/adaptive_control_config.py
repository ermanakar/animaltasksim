"""Configuration for the adaptive control agent family."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from agents.losses import LossWeights
from animaltasksim.config import ProjectPaths


AdaptiveControlProfile: TypeAlias = Literal[
    "no_control",
    "persistence_only",
    "exploration_only",
    "full_control",
]
ResolvedAdaptiveControlProfile: TypeAlias = Literal[
    "no_control",
    "persistence_only",
    "exploration_only",
    "full_control",
    "custom",
]

RECOMMENDED_ADAPTIVE_CONTROL_PROFILE: AdaptiveControlProfile = "persistence_only"

ADAPTIVE_CONTROL_PROFILE_FLAGS: dict[AdaptiveControlProfile, dict[str, bool]] = {
    "no_control": {
        "control_state_enabled": False,
        "persistence_enabled": False,
        "exploration_enabled": False,
    },
    "persistence_only": {
        "control_state_enabled": True,
        "persistence_enabled": True,
        "exploration_enabled": False,
    },
    "exploration_only": {
        "control_state_enabled": True,
        "persistence_enabled": False,
        "exploration_enabled": True,
    },
    "full_control": {
        "control_state_enabled": True,
        "persistence_enabled": True,
        "exploration_enabled": True,
    },
}


def adaptive_control_profile_flags(profile: AdaptiveControlProfile) -> dict[str, bool]:
    """Return control booleans for a named adaptive-control profile."""
    return dict(ADAPTIVE_CONTROL_PROFILE_FLAGS[profile])


@dataclass(slots=True)
class AdaptiveControlPaths:
    """Collection of output artifact locations."""

    root: Path
    config: Path
    log: Path
    metrics: Path
    model: Path


@dataclass(slots=True)
class AdaptiveControlConfig:
    """Training and rollout configuration for the adaptive control agent.

    Args:
        control_state_enabled: Enables the adaptive-control fast state. Disable
            this for the clean no-control lesion.
        persistence_enabled: Enables the validated persistence/retry controller.
            This stays on in the recommended persistence-only default.
        exploration_enabled: Enables the rewarded-streak/staleness exploration
            controller. Defaults to False because phase-1 exploration was not
            independently validated; turn it on only for experimental
            exploration-only or full-control comparisons.
    """

    task: Literal["ibl_2afc", "rdm"] = "ibl_2afc"
    reference_log: Path | None = None
    output_dir: Path | None = None
    agent_version: str = "0.1.0"
    trials_per_episode: int = 400
    episodes: int = 10
    seed: int = 1234
    epochs: int = 5
    hidden_size: int = 64
    learning_rate: float = 1e-3
    loss_weights: LossWeights = field(default_factory=LossWeights)
    step_ms: int = 10
    max_sessions: int | None = None
    max_trials_per_session: int | None = None
    min_commit_steps: int = 5
    max_commit_steps: int = 300
    drift_scale: float = 6.0
    drift_magnitude_target: float = 12.0
    history_bias_scale: float = 2.0
    history_drift_scale: float = 0.3
    lapse_rate: float = 0.05
    freeze_history_scales: bool = False
    inject_win_tendency: float | None = None
    inject_lose_tendency: float | None = None
    anneal_history_injection: bool = False
    history_injection_alpha_start: float = 1.0
    history_injection_alpha_end: float = 0.0
    control_state_enabled: bool = True
    persistence_enabled: bool = True
    exploration_enabled: bool = False
    persistence_learning_rate: float = 0.8
    switch_learning_rate: float = 0.8
    reward_learning_rate: float = 0.6
    control_state_decay: float = 0.7
    control_state_scale: float = 1.0
    persistence_bias_scale: float = 1.6
    exploration_bias_scale: float = 0.8
    control_residual_limit: float = 0.35
    control_pressure_limit: float = 0.35
    control_uncertainty_power: float = 2.0
    adaptive_control_regularization: float = 0.05
    evidence_preservation_regularization: float = 0.05

    def __post_init__(self) -> None:
        self.control_residual_limit = max(0.0, float(self.control_residual_limit))
        self.control_pressure_limit = max(0.0, float(self.control_pressure_limit))
        self.control_uncertainty_power = max(1.0, float(self.control_uncertainty_power))
        self.adaptive_control_regularization = max(0.0, float(self.adaptive_control_regularization))
        self.evidence_preservation_regularization = max(
            0.0,
            float(self.evidence_preservation_regularization),
        )
        paths = ProjectPaths.from_cwd()
        if self.reference_log is None:
            if self.task == "ibl_2afc":
                self.reference_log = paths.data / "ibl" / "reference.ndjson"
            else:
                self.reference_log = paths.data / "macaque" / "reference.ndjson"
        if self.output_dir is None:
            if self.task == "ibl_2afc":
                self.output_dir = paths.runs / "ibl_adaptive_control"
            else:
                self.output_dir = paths.runs / "rdm_adaptive_control"

    @property
    def active_control_profile(self) -> ResolvedAdaptiveControlProfile:
        """Return the named profile matching the current control switches."""
        flags = {
            "control_state_enabled": self.control_state_enabled,
            "persistence_enabled": self.persistence_enabled,
            "exploration_enabled": self.exploration_enabled,
        }
        for profile, profile_flags in ADAPTIVE_CONTROL_PROFILE_FLAGS.items():
            if flags == profile_flags:
                return profile
        return "custom"

    def apply_control_profile(self, profile: AdaptiveControlProfile) -> None:
        """Apply a named lesion/comparison profile to control switches."""
        flags = adaptive_control_profile_flags(profile)
        self.control_state_enabled = flags["control_state_enabled"]
        self.persistence_enabled = flags["persistence_enabled"]
        self.exploration_enabled = flags["exploration_enabled"]

    @classmethod
    def recommended(cls, **overrides: object) -> AdaptiveControlConfig:
        """Build the recommended persistence-only adaptive-control config."""
        config = cls(**overrides)
        config.apply_control_profile(RECOMMENDED_ADAPTIVE_CONTROL_PROFILE)
        return config

    @classmethod
    def for_control_profile(
        cls,
        profile: AdaptiveControlProfile,
        **overrides: object,
    ) -> AdaptiveControlConfig:
        """Build a config for a named lesion/comparison profile."""
        config = cls(**overrides)
        config.apply_control_profile(profile)
        return config

    def output_paths(self) -> AdaptiveControlPaths:
        out = Path(self.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return AdaptiveControlPaths(
            root=out,
            config=out / "config.json",
            log=out / "trials.ndjson",
            metrics=out / "training_metrics.json",
            model=out / "model.pt",
        )


__all__ = [
    "ADAPTIVE_CONTROL_PROFILE_FLAGS",
    "RECOMMENDED_ADAPTIVE_CONTROL_PROFILE",
    "AdaptiveControlConfig",
    "AdaptiveControlPaths",
    "AdaptiveControlProfile",
    "ResolvedAdaptiveControlProfile",
    "adaptive_control_profile_flags",
]
