"""Configuration for the adaptive control agent family."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents.losses import LossWeights
from animaltasksim.config import ProjectPaths


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
    """Training and rollout configuration for the adaptive control agent."""

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
    drift_scale: float = 10.0
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
    persistence_enabled: bool = True
    exploration_enabled: bool = True
    persistence_learning_rate: float = 0.8
    switch_learning_rate: float = 0.8
    reward_learning_rate: float = 0.6
    control_state_decay: float = 0.7
    control_state_scale: float = 1.0
    persistence_bias_scale: float = 0.8
    exploration_bias_scale: float = 0.8

    def __post_init__(self) -> None:
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


__all__ = ["AdaptiveControlConfig", "AdaptiveControlPaths"]
