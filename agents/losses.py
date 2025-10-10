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
    history: float = 0.0

    def clamp_non_negative(self) -> None:
        """Ensure weights remain non-negative."""
        self.choice = max(0.0, float(self.choice))
        self.rt = max(0.0, float(self.rt))
        self.history = max(0.0, float(self.history))


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

    loss = ((predicted - targets) / 1000.0) ** 2
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


__all__ = ["LossWeights", "choice_loss", "rt_loss", "history_penalty"]
