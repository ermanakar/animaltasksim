"""Differentiable finite-window DDM choice and reported response-time surrogate."""
from __future__ import annotations

import torch

__all__ = ["soft_first_passage"]


def soft_first_passage(
    evidence_trajectory: torch.Tensor,
    bound: torch.Tensor,
    non_decision_ms: torch.Tensor,
    *,
    step_ms: float,
    min_commit_steps: int,
    max_commit_steps: int,
    lapse_rate: float = 0.0,
    temperature: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return right probability and expected quantized RT for supplied paths.

    The last dimension is the sequence of post-increment evidence positions.
    Other tensors broadcast over the leading dimensions. Soft boundary hazards
    approximate hard first crossing; surviving paths choose by final sign at
    timeout. RT uses the expectation of clipped, quantized individual outcomes,
    including nondecision time and the rollout's uniform-step lapse mixture.

    Quantization uses a straight-through gradient for nondecision time: forward
    values match floor exactly, while its backward derivative is one. This is
    a training surrogate, not a differentiable exact first-passage likelihood.
    A final evidence value exactly zero favors left, matching hard rollout.
    """
    if evidence_trajectory.ndim < 1 or evidence_trajectory.shape[-1] != max_commit_steps:
        raise ValueError("Trajectory length must equal max_commit_steps")
    if not 1 <= min_commit_steps <= max_commit_steps:
        raise ValueError("Require 1 <= min_commit_steps <= max_commit_steps")
    if step_ms <= 0 or temperature <= 0 or not 0 <= lapse_rate <= 1:
        raise ValueError("Require positive step_ms/temperature and lapse_rate in [0, 1]")

    upper = torch.sigmoid((evidence_trajectory - bound.unsqueeze(-1)) / temperature)
    lower = torch.sigmoid((-bound.unsqueeze(-1) - evidence_trajectory) / temperature)
    hazard = (upper + lower).clamp(max=1.0)
    survival = torch.cumprod(1.0 - hazard, dim=-1)
    survival_before = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), dim=-1)
    mass = survival_before * hazard
    timeout = survival[..., -1]
    final = evidence_trajectory[..., -1]
    timeout_right = torch.where(final == 0, torch.zeros_like(final), torch.sigmoid(final / temperature))
    p_right = (survival_before * upper).sum(dim=-1) + timeout * timeout_right

    steps = torch.arange(1, max_commit_steps + 1, dtype=final.dtype, device=final.device)
    response_steps = steps + non_decision_ms.unsqueeze(-1) / step_ms
    quantized = response_steps + (torch.floor(response_steps) - response_steps).detach()
    reported_ms = quantized.clamp(min=min_commit_steps, max=max_commit_steps) * step_ms
    mean_rt_ms = (mass * reported_ms).sum(dim=-1) + timeout * reported_ms[..., -1]
    lapse_rt = reported_ms[..., min_commit_steps - 1:].mean(dim=-1)
    return (
        (1.0 - lapse_rate) * p_right + lapse_rate * 0.5,
        (1.0 - lapse_rate) * mean_rt_ms + lapse_rate * lapse_rt,
    )
