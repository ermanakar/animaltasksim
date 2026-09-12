"""Calibration of the finite-window soft surrogate against hard first passage."""
from __future__ import annotations

import pytest
import torch

from agents.ddm_simulation import soft_first_passage


def _hard(paths: torch.Tensor, bound: float, ndt: float, step_ms: float, minimum: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent hard first crossing and rollout RT mapping on fixed paths."""
    right = []
    rts = []
    maximum = paths.shape[-1]
    for path in paths:
        choice = float(path[-1] > 0)
        step = maximum
        for i, evidence in enumerate(path):
            if abs(evidence) >= bound:
                choice = float(evidence >= bound)
                step = i + 1
                break
        right.append(choice)
        rts.append(min(max(int((ndt + step * step_ms) / step_ms), minimum), maximum) * step_ms)
    return torch.tensor(right), torch.tensor(rts)


def test_decreasing_temperature_converges_on_supplied_paths() -> None:
    """Crossings, recrossings, timeouts and zero-final ties match the hard rule."""
    paths = torch.tensor([[.2, 1.2, -.3, -1.4], [-.4, -.6, -1.3, .8],
                          [.1, .2, .3, .4], [.2, -.1, -.2, 0.]])
    expected_p, expected_rt = _hard(paths, 1., 15., 10., 1)
    errors = []
    for temp in [.5, .1, .01]:
        p, rt = soft_first_passage(paths, torch.tensor(1.), torch.tensor(15.),
                                   step_ms=10, min_commit_steps=1, max_commit_steps=4, temperature=temp)
        errors.append((p - expected_p).abs().mean() + (rt - expected_rt).abs().mean() / 40)
    assert errors[-1] < errors[1] < errors[0]
    assert torch.allclose(p, expected_p, atol=1e-6)
    assert torch.allclose(rt, expected_rt, atol=1e-5)


def test_lapse_uses_uniform_integer_steps_and_quantization() -> None:
    """Lapse RT averages mapped outcomes, with nondecision clipping per outcome."""
    p, rt = soft_first_passage(torch.zeros(5), torch.tensor(1.), torch.tensor(15.),
                               step_ms=10, min_commit_steps=2, max_commit_steps=5, lapse_rate=1.)
    assert p.item() == .5
    assert rt.item() == pytest.approx((30 + 40 + 50 + 50) / 4)


def test_gradients_are_finite_and_ndt_uses_straight_through_quantization() -> None:
    paths = torch.tensor([[.2, .995, 1.02, 1.3]], requires_grad=True)
    bound = torch.tensor(1., requires_grad=True)
    ndt = torch.tensor(1., requires_grad=True)
    p, rt = soft_first_passage(paths, bound, ndt, step_ms=10, min_commit_steps=1, max_commit_steps=4)
    (p.sum() + rt.sum()).backward()
    for value in [paths, bound, ndt]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize("temperature", [.1, .01])
@pytest.mark.parametrize("drift,bound,noise", [(-1., .5, 1.), (0., 1., 1.), (2., 1.5, .5)])
def test_soft_stochastic_calibration(drift: float, bound: float, noise: float, temperature: float) -> None:
    """Bound approximation error on paired seeded trajectories, not fresh samples."""
    generator = torch.Generator().manual_seed(81)
    paths = (drift * .01 + noise * .1 * torch.randn(3000, 150, generator=generator)).cumsum(-1)
    p_hard, rt_hard = _hard(paths, bound, 150., 10., 5)
    p_soft, rt_soft = soft_first_passage(paths, torch.tensor(bound), torch.tensor(150.),
                                        step_ms=10, min_commit_steps=5, max_commit_steps=150, temperature=temperature)
    # Fixed temperature smooths near misses into early crossings. These bounds
    # document that approximation; they are not claims of likelihood equivalence.
    assert abs(p_soft.mean() - p_hard.mean()) < (.1 if temperature == .1 else .005)
    assert abs(rt_soft.mean() - rt_hard.mean()) < (250 if temperature == .1 else 10)
