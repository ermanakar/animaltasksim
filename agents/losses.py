"""Shared loss utilities for agents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class LossWeights:
    """Weights for multi-objective behavioural training."""

    choice: float = 1.0
    rt: float = 1.0
    rt_soft: float = 0.0
    history: float = 0.0
    drift_supervision: float = 0.0
    non_decision_supervision: float = 0.0
    wfpt: float = 0.0  # Wiener First Passage Time likelihood loss
    drift_magnitude: float = 0.0  # Regularization to anchor drift_gain scale
    twin_supervision: float = 0.0  # Encourage alignment with per-session DDM fits

    def clamp_non_negative(self) -> None:
        """Ensure weights remain non-negative."""
        self.choice = max(0.0, float(self.choice))
        self.rt = max(0.0, float(self.rt))
        self.rt_soft = max(0.0, float(self.rt_soft))
        self.history = max(0.0, float(self.history))
        self.drift_supervision = max(0.0, float(self.drift_supervision))
        self.non_decision_supervision = max(0.0, float(self.non_decision_supervision))
        self.wfpt = max(0.0, float(self.wfpt))
        self.drift_magnitude = max(0.0, float(self.drift_magnitude))
        self.twin_supervision = max(0.0, float(self.twin_supervision))


def choice_loss(probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Binary cross-entropy on right-choice probability.

    Parameters
    ----------
    probs:
        Probability of choosing right (values in [0, 1]).
    targets:
        Ground-truth choice (1 for right, 0 for left). Hold trials should set mask=0.
    mask:
        Optional mask tensor (0 to skip, 1 to include).
    """

    probs = torch.clamp(probs, 1e-5, 1.0 - 1e-5)
    loss = F.binary_cross_entropy(probs, targets, reduction="none")
    if mask is not None:
        loss = loss * mask
    normaliser = loss.numel() if mask is None else torch.clamp(mask.sum(), min=1.0)
    return loss.sum() / normaliser


def rt_loss(predicted: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-squared error loss on reaction times (scaled to seconds)."""
    loss = (predicted - targets) ** 2
    if mask is not None:
        loss = loss * mask
    normaliser = loss.numel() if mask is None else torch.clamp(mask.sum(), min=1.0)
    return loss.sum() / normaliser


def history_penalty(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Quadratic penalty for history statistics (win-stay / lose-shift).

    Both `pred` and `target` are expected to be aggregated statistics, so the
    mask defaults to a tensor of ones.
    """

    if mask is None:
        mask = torch.ones_like(pred)
    loss = ((pred - target) ** 2) * mask
    normaliser = torch.clamp(mask.sum(), min=1.0)
    return loss.sum() / normaliser


def drift_supervision_loss(drift_gain: torch.Tensor, target_gain: float = 5.0) -> torch.Tensor:
    """Penalize drift_gain parameters that are too weak to produce RT dynamics.
    
    Parameters
    ----------
    drift_gain:
        Predicted drift_gain values from the model.
    target_gain:
        Target drift_gain magnitude (default 5.0 for reasonable SNR at high coherence).
    
    Returns
    -------
    loss:
        Quadratic penalty for drift_gain values below target.
    """
    # Only penalize if drift_gain is below target (allow it to be higher)
    loss = torch.clamp(target_gain - drift_gain, min=0.0) ** 2
    return loss.mean()


def non_decision_supervision_loss(
    non_decision_ms: torch.Tensor, target_ms: float = 200.0
) -> torch.Tensor:
    """Penalize non-decision times that are far from a target value."""
    loss = (non_decision_ms - target_ms) ** 2
    return loss.mean()


def soft_rt_penalty(
    predicted_rt: torch.Tensor,
    target_rt: torch.Tensor,
    target_var: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft penalty for reaction times toward per-coherence means with variance scaling."""

    loss = ((predicted_rt - target_rt) ** 2) / torch.clamp(target_var, min=1e-3)
    if mask is not None:
        loss = loss * mask
    normaliser = loss.numel() if mask is None else torch.clamp(mask.sum(), min=1.0)
    return loss.sum() / normaliser


__all__ = [
    "LossWeights",
    "choice_loss",
    "rt_loss",
    "soft_rt_penalty",
    "history_penalty",
    "drift_supervision_loss",
    "non_decision_supervision_loss",
]
