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
    history_supervision: float = 0.0  # MSE loss on win-stay/lose-shift targets
    per_trial_history: float = 0.0  # Per-trial differentiable history loss

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
        self.history_supervision = max(0.0, float(self.history_supervision))
        self.per_trial_history = max(0.0, float(self.per_trial_history))


def history_supervision_loss(
    pred_win_stay: torch.Tensor,
    pred_lose_shift: torch.Tensor,
    target_win_stay: torch.Tensor,
    target_lose_shift: torch.Tensor,
) -> torch.Tensor:
    """MSE loss to pull win-stay/lose-shift probabilities toward macaque values."""
    loss_ws = F.mse_loss(pred_win_stay, target_win_stay)
    loss_ls = F.mse_loss(pred_lose_shift, target_lose_shift)
    return (loss_ws + loss_ls) / 2.0


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
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft penalty for reaction times toward per-coherence means with variance scaling."""

    loss = ((predicted_rt - target_rt) ** 2) / torch.clamp(target_var, min=1e-3)
    if weights is not None:
        loss = loss * weights
    if mask is not None:
        loss = loss * mask
    normaliser = loss.numel() if mask is None else torch.clamp(mask.sum(), min=1.0)
    return loss.sum() / normaliser


def per_trial_history_loss(
    choice_prob: torch.Tensor,
    prev_action: torch.Tensor,
    prev_reward: torch.Tensor,
    target_win_stay: float = 0.72,
    target_lose_shift: float = 0.43,
    mask: torch.Tensor | None = None,
    no_action_value: float = -1.0,
) -> torch.Tensor:
    """Per-trial differentiable history loss targeting win-stay/lose-shift.

    Unlike batch-aggregate history regularisers that constrain only the mean,
    this loss supervises *every trial individually*, giving per-trial gradient
    signal to the recurrent hidden state.  This closes the "decoupling gap"
    where models learn chronometric slopes but fail to capture inter-trial
    history effects.

    Args:
        choice_prob: P(right) per trial, must retain gradient.
        prev_action: Previous action.  Convention: 1 = right,
            0 or -1 = left depending on encoding.  The sentinel for "no
            valid previous action" is given by *no_action_value*.
        prev_reward: Previous reward (>0.5 → win, ≤0.5 → loss).
        target_win_stay: Animal-derived P(stay | prev win).
        target_lose_shift: Animal-derived P(shift | prev loss).
        mask: Optional validity mask (1=include, 0=skip).
        no_action_value: Sentinel encoding "no previous action".
            R-DDM uses -1 (default); Hybrid uses 0.

    Returns:
        Scalar MSE loss averaged over valid trials.
    """
    # P(stay) = P(same action as previous)
    # prev_action == 1 means "previous was right" in both conventions
    stay_prob = torch.where(
        prev_action == 1,
        choice_prob,
        1.0 - choice_prob,
    )

    valid = prev_action.ne(no_action_value)
    if mask is not None:
        valid = valid & mask

    win_mask = valid & (prev_reward > 0.5)
    lose_mask = valid & (prev_reward <= 0.5)

    loss = torch.zeros(1, device=choice_prob.device, dtype=choice_prob.dtype)

    if win_mask.any():
        # Win trials: push P(stay) toward target_win_stay per trial
        target = torch.full_like(stay_prob[win_mask], target_win_stay)
        loss = loss + F.mse_loss(stay_prob[win_mask], target)

    if lose_mask.any():
        # Lose trials: push P(shift)=1-P(stay) toward target_lose_shift per trial
        shift_prob = 1.0 - stay_prob
        target = torch.full_like(shift_prob[lose_mask], target_lose_shift)
        loss = loss + F.mse_loss(shift_prob[lose_mask], target)

    return loss.squeeze()


__all__ = [
    "LossWeights",
    "choice_loss",
    "rt_loss",
    "soft_rt_penalty",
    "history_penalty",
    "history_supervision_loss",
    "drift_supervision_loss",
    "non_decision_supervision_loss",
    "per_trial_history_loss",
]
