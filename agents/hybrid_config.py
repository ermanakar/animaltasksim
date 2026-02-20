"""Data structures and configuration for the hybrid agent."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from agents.losses import LossWeights
from agents.curriculum import CurriculumConfig
from animaltasksim.config import ProjectPaths

@dataclass(slots=True)
class HybridDDMPaths:
    """Collection of output artefact locations."""

    root: Path
    config: Path
    log: Path
    metrics: Path
    model: Path
@dataclass(slots=True)
class HybridTrainingConfig:
    """Training and rollout configuration for the hybrid agent."""

    task: str = "rdm"  # "rdm" or "ibl_2afc"
    reference_log: Path = field(default=None)  # type: ignore[assignment]  # Set in __post_init__
    output_dir: Path = field(default=None)  # type: ignore[assignment]  # Set in __post_init__
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
    max_commit_steps: int = 300  # Must accommodate DDM boundary crossings at low coherence
    drift_scale: float = 10.0  # Scale drift_head initialization to enable stronger evidence effects
    curriculum: CurriculumConfig | None = None  # If set, use curriculum learning
    history_bias_scale: float = 0.5  # History bias can shift starting point by ±scale*bound
    history_drift_scale: float = 0.0  # History bias can add ±scale to drift rate (0=disabled)

    def __post_init__(self) -> None:
        paths = ProjectPaths.from_cwd()
        if self.reference_log is None:
            if self.task == "ibl_2afc":
                self.reference_log = paths.data / "ibl" / "reference.ndjson"
            else:
                self.reference_log = paths.data / "macaque" / "reference.ndjson"
        if self.output_dir is None:
            if self.task == "ibl_2afc":
                self.output_dir = paths.runs / "ibl_hybrid"
            else:
                self.output_dir = paths.runs / "rdm_hybrid"

    def output_paths(self) -> HybridDDMPaths:
        out = Path(self.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return HybridDDMPaths(
            root=out,
            config=out / "config.json",
            log=out / "trials.ndjson",
            metrics=out / "training_metrics.json",
            model=out / "model.pt",
        )
@dataclass(slots=True)
class SessionBatch:
    """Reference data for a single animal session."""

    features: np.ndarray  # shape (T, F)
    choice: np.ndarray  # shape (T,)
    choice_mask: np.ndarray  # shape (T,)
    rt_ms: np.ndarray  # shape (T,)
    rt_mask: np.ndarray  # shape (T,)
    correct: np.ndarray  # shape (T,)
    win_stay_target: float
    lose_shift_target: float
    twin_params: dict[str, float] | None = None
    rt_targets: np.ndarray | None = None
    rt_variances: np.ndarray | None = None
