"""Configuration objects for the Recurrent Drift-Diffusion Model (R-DDM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class RDDMTrainingSchedule:
    """Curriculum weights for the individual loss terms."""

    warmup_epochs: int = 6
    """Number of initial epochs that emphasise the choice likelihood."""

    choice_warmup_weight: float = 3.0
    """Multiplier applied to the choice loss during the warm-up phase."""

    enable_wfpt_epoch: int = 6
    wfpt_ramp_epochs: int = 5
    enable_history_epoch: int = 10
    """Epoch index (0-based) at which the history regulariser becomes active."""


@dataclass(slots=True)
class RDDMConfig:
    """Top-level configuration for training the R-DDM agent."""

    # Data
    task: Literal["ibl_2afc", "rdm_macaque"] = "ibl_2afc"
    reference_log: Path | None = None
    """Path to the multi-session IBL reference NDJSON log."""

    max_sessions: int | None = None
    """Optional cap on the number of sessions to load (useful for smoke tests)."""

    # Model
    input_features: Literal[8] = 8
    """Number of per-trial input features consumed by the model."""

    hidden_size: int = 32
    """Hidden size of the GRU that aggregates trial history."""

    num_layers: int = 1
    """Number of stacked GRU layers."""

    dropout: float = 0.0
    """Dropout probability applied between GRU layers (0.0 disables it)."""

    # Training
    batch_size: int = 8
    """Number of sessions per optimisation step."""

    learning_rate: float = 1e-3
    """Adam learning rate."""

    weight_decay: float = 0.0
    """Optional L2 penalty applied by the optimiser."""

    gradient_clip: float = 1.0
    """Maximum gradient norm (L2) before clipping."""

    epochs: int = 20
    """Total number of passes over the dataset."""

    schedule: RDDMTrainingSchedule = field(default_factory=RDDMTrainingSchedule)
    """Curriculum controlling the activation of secondary loss terms."""

    # Loss weights
    choice_loss_weight: float = 1.0
    wfpt_loss_weight: float = 0.5
    history: float = 1.0
    side_bias: float = 1.0
    per_trial_history_weight: float = 0.5
    """Weight for per-trial differentiable history loss (0 disables)."""
    drift_supervision_weight: float = 0.2
    """Weight for the signed drift supervision loss."""

    choice_kl_weight: float = 0.0
    """Weight for the KL divergence between analytic stimulus target and predicted choice prob."""

    choice_kl_target_slope: float = 6.0
    """Slope applied to the logistic target used by the KL regulariser."""

    entropy_loss_weight: float = 0.0
    entropy_target: float = 0.35  # Target entropy in nats
    entropy_weight_lr: float = 1e-3  # Learning rate for entropy weight
    freeze_bias_epochs: int = 3
    """Weight for the entropy penalty encouraging confident categorical predictions."""

    # History targets (from ibl_reference_metrics.json)
    target_win_stay: float = 0.7246745473589705
    target_lose_shift: float = 0.43374197314652657
    
    # Parameter ranges (seconds unless noted)
    min_boundary: float = 0.5
    max_boundary: float = 3.0

    min_non_decision: float = 0.05
    max_non_decision: float = 0.35

    min_noise: float = 0.5
    max_noise: float = 1.5

    max_drift: float = 6.0

    non_decision_target: float = 0.3
    """Target non-decision time (seconds) used for regularisation."""

    non_decision_reg_weight: float = 0.1
    history_ramp_epochs: int = 8

    # Feature scaling
    prior_feature_scale: float = 0.2
    history_feature_scale: float = 0.5
    stimulus_scale: float = 1.0

    # Runtime
    seed: int = 42
    device: str = "cpu"
    rollout_trials: int = 1200
    motor_delay_ms: float = 0.0
    freeze_bias_epochs: int = 0
    """Additional motor delay added during rollouts (ms)."""

    # Logging / output
    run_dir: Path = Path("runs/r_ddm_experiment")
    """Directory where training artefacts and rollouts are written."""

    def __post_init__(self) -> None:
        if self.reference_log is None:
            if self.task == "rdm_macaque":
                self.reference_log = Path("data/macaque/reference.ndjson")
            else:
                self.reference_log = Path("data/ibl/reference.ndjson")
        if self.task == "rdm_macaque":
            if self.target_win_stay == 0.7246745473589705:
                self.target_win_stay = 0.49
            if self.target_lose_shift == 0.43374197314652657:
                self.target_lose_shift = 0.516
            self.prior_feature_scale = 0.0 if self.prior_feature_scale == 0.2 else self.prior_feature_scale
            if self.stimulus_scale == 1.0:
                self.stimulus_scale = 0.512
        else:
            if self.stimulus_scale == 1.0:
                self.stimulus_scale = 1.0
        if self.task == "rdm_macaque" and self.motor_delay_ms == 200.0:
            self.motor_delay_ms = 120.0
