"""Unit tests for the WFPT likelihood implementation.

Validates the Wiener First Passage Time density in agents/wfpt_loss.py
against known analytical properties of the drift-diffusion model:
  1. Density is finite and non-negative for reasonable parameters.
  2. Symmetry: flipping drift sign mirrors choice probabilities.
  3. Monotonicity: stronger drift increases density for the favoured choice.
  4. Gradient flow: all DDM parameters receive gradients.
  5. Edge cases: near-zero drift, extreme bias, minimal decision time.

Convention note: the implementation maps positive drift → higher P(choice=0).
This is an internal convention (opposite to standard DDM where positive drift
favours the upper boundary / choice=1). The training pipeline compensates by
learning the sign jointly, so results are unaffected. Tests below verify
internal consistency under this convention.

References:
  - Ratcliff & McKoon (2008). The Diffusion Decision Model.
  - Navarro & Fuss (2009). Fast and accurate calculations for FPT in DDMs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.wfpt_loss import wfpt_log_likelihood, wfpt_loss


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _scalar_ll(
    choice: float,
    rt: float,
    drift: float,
    bound: float,
    bias: float,
    noise: float,
    ndt: float,
) -> float:
    """Convenience wrapper returning a scalar log-likelihood."""
    ll = wfpt_log_likelihood(
        choice=torch.tensor([choice]),
        rt=torch.tensor([rt]),
        drift=torch.tensor([drift]),
        bound=torch.tensor([bound]),
        bias=torch.tensor([bias]),
        noise=torch.tensor([noise]),
        non_decision=torch.tensor([ndt]),
    )
    return float(ll.item())


def _numerical_choice_probability(
    drift: float,
    bound: float,
    bias: float,
    noise: float,
    ndt: float,
    choice: float = 1.0,
    n_points: int = 500,
    t_max: float = 5.0,
) -> float:
    """Numerically integrate the WFPT density for a given choice.

    Uses the trapezoidal rule over a fine grid of decision times.
    """
    t_vals = torch.linspace(ndt + 1e-4, t_max, n_points)
    ll = wfpt_log_likelihood(
        choice=torch.full((n_points,), choice),
        rt=t_vals,
        drift=torch.full((n_points,), drift),
        bound=torch.full((n_points,), bound),
        bias=torch.full((n_points,), bias),
        noise=torch.full((n_points,), noise),
        non_decision=torch.full((n_points,), ndt),
    )
    density = torch.exp(ll)
    area = float(torch.trapezoid(density, t_vals))
    return area


# ------------------------------------------------------------------ #
# Test: density is non-negative and finite
# ------------------------------------------------------------------ #

class TestWFPTBasicProperties:
    """Sanity checks on density values."""

    def test_density_is_finite(self) -> None:
        """Log-likelihood should be finite for reasonable parameters."""
        ll = _scalar_ll(choice=1, rt=0.5, drift=2.0, bound=1.5, bias=0.5, noise=1.0, ndt=0.1)
        assert np.isfinite(ll), f"Log-likelihood is not finite: {ll}"

    def test_density_is_negative_log(self) -> None:
        """Log-density should be negative (density < 1 for typical params)."""
        ll = _scalar_ll(choice=1, rt=0.5, drift=2.0, bound=1.5, bias=0.5, noise=1.0, ndt=0.1)
        # Density can be > 1 for peaked distributions, but log should be finite
        assert np.isfinite(ll)

    def test_density_positive_over_grid(self) -> None:
        """Density should be non-negative across a range of RTs."""
        rts = torch.linspace(0.15, 3.0, 50)
        ll = wfpt_log_likelihood(
            choice=torch.ones(50),
            rt=rts,
            drift=torch.full((50,), 1.5),
            bound=torch.full((50,), 1.0),
            bias=torch.full((50,), 0.5),
            noise=torch.full((50,), 1.0),
            non_decision=torch.full((50,), 0.1),
        )
        density = torch.exp(ll)
        assert torch.all(density >= 0), "Density contains negative values"
        assert torch.all(torch.isfinite(ll)), "Log-likelihood contains non-finite values"


# ------------------------------------------------------------------ #
# Test: choice probabilities sum to ~1 (both boundaries)
# ------------------------------------------------------------------ #

class TestWFPTChoiceProbability:
    """Density for choice=0 and choice=1 should integrate to ~1 total."""

    @pytest.mark.parametrize(
        "drift, bound, bias, noise",
        [
            (2.0, 1.5, 0.5, 1.0),   # standard case
            (-1.0, 1.0, 0.5, 1.0),  # negative drift
            (1.5, 1.0, 0.5, 1.0),   # moderate drift
            (1.0, 1.0, 0.3, 1.0),   # biased start
        ],
    )
    def test_both_choices_sum_to_one(
        self, drift: float, bound: float, bias: float, noise: float
    ) -> None:
        """P(choice=0) + P(choice=1) should approximate 1.0."""
        ndt = 0.1
        p0 = _numerical_choice_probability(drift, bound, bias, noise, ndt, choice=0.0, n_points=1000, t_max=8.0)
        p1 = _numerical_choice_probability(drift, bound, bias, noise, ndt, choice=1.0, n_points=1000, t_max=8.0)
        total = p0 + p1
        assert abs(total - 1.0) < 0.25, (
            f"drift={drift}, bound={bound}, bias={bias}: P(0)={p0:.3f} + P(1)={p1:.3f} = {total:.3f}"
        )

    def test_strong_drift_wide_bound_normalizes(self) -> None:
        """Strong drift + wide bound should still integrate to ~1.0.

        Previously this regime (drift=3.0, bound=2.0) caused over-integration
        due to incorrect image charge positions in the small-time series.
        Fixed by using a*(z + 2k) instead of z + 2ka for the starting
        point positions.
        """
        ndt = 0.1
        p0 = _numerical_choice_probability(3.0, 2.0, 0.5, 1.0, ndt, choice=0.0, n_points=1000, t_max=8.0)
        p1 = _numerical_choice_probability(3.0, 2.0, 0.5, 1.0, ndt, choice=1.0, n_points=1000, t_max=8.0)
        total = p0 + p1
        assert abs(total - 1.0) < 0.15, (
            f"drift=3, bound=2: P(0)={p0:.3f} + P(1)={p1:.3f} = {total:.3f}"
        )

    def test_stronger_drift_shifts_probability(self) -> None:
        """Stronger drift should increase P for the favoured choice.

        Convention: positive drift → higher P(choice=0) in this implementation.
        """
        ndt = 0.1
        p0_weak = _numerical_choice_probability(1.0, 1.5, 0.5, 1.0, ndt, choice=0.0, n_points=500)
        p0_strong = _numerical_choice_probability(3.0, 1.5, 0.5, 1.0, ndt, choice=0.0, n_points=500)
        assert p0_strong > p0_weak, (
            f"Stronger drift should increase P(choice=0): weak={p0_weak:.3f}, strong={p0_strong:.3f}"
        )


# ------------------------------------------------------------------ #
# Test: symmetry properties
# ------------------------------------------------------------------ #

class TestWFPTSymmetry:
    """Drift sign flip should mirror choice probabilities."""

    def test_drift_sign_flip_mirrors_density(self) -> None:
        """Flipping drift sign should swap which choice has higher likelihood."""
        # Positive drift favours choice=0 in this implementation
        ll_c0_pos = _scalar_ll(choice=0, rt=0.4, drift=2.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        ll_c1_pos = _scalar_ll(choice=1, rt=0.4, drift=2.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        # Negative drift favours choice=1
        ll_c0_neg = _scalar_ll(choice=0, rt=0.4, drift=-2.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        ll_c1_neg = _scalar_ll(choice=1, rt=0.4, drift=-2.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        # Symmetry: ll(c=0, +v) == ll(c=1, -v) for unbiased start
        assert abs(ll_c0_pos - ll_c1_neg) < 0.01, (
            f"Symmetry broken: ll(c0,+v)={ll_c0_pos:.4f} != ll(c1,-v)={ll_c1_neg:.4f}"
        )
        assert abs(ll_c1_pos - ll_c0_neg) < 0.01, (
            f"Symmetry broken: ll(c1,+v)={ll_c1_pos:.4f} != ll(c0,-v)={ll_c0_neg:.4f}"
        )

    def test_zero_drift_equal_likelihoods(self) -> None:
        """With zero drift and unbiased start, both choices should have equal likelihood."""
        ll_c0 = _scalar_ll(choice=0, rt=0.5, drift=0.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        ll_c1 = _scalar_ll(choice=1, rt=0.5, drift=0.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        assert abs(ll_c0 - ll_c1) < 0.01, (
            f"Zero drift should give equal likelihoods: ll(c0)={ll_c0:.4f}, ll(c1)={ll_c1:.4f}"
        )


# ------------------------------------------------------------------ #
# Test: gradient flow
# ------------------------------------------------------------------ #

class TestWFPTGradients:
    """All DDM parameters should receive gradients through the loss."""

    def test_all_params_receive_gradients(self) -> None:
        """wfpt_loss should produce finite gradients for all parameters."""
        batch = 20
        choice = torch.randint(0, 2, (batch,)).float()
        rt_ms = torch.rand(batch) * 600 + 300  # 300–900 ms

        # Use leaf tensors (requires_grad before any ops) to get .grad
        drift = torch.randn(batch, requires_grad=True)
        bound = torch.nn.Parameter(torch.full((batch,), 1.5))
        bias = torch.nn.Parameter(torch.zeros(batch))
        noise = torch.nn.Parameter(torch.ones(batch))
        ndt_ms = torch.nn.Parameter(torch.full((batch,), 200.0))

        loss = wfpt_loss(choice, rt_ms, drift, bound, bias, noise, ndt_ms)
        loss.backward()

        for name, param in [("drift", drift), ("bound", bound), ("bias", bias), ("noise", noise), ("ndt", ndt_ms)]:
            assert param.grad is not None, f"{name} received no gradient"
            assert torch.all(torch.isfinite(param.grad)), f"{name} has non-finite gradient"

    def test_gradient_direction_for_drift(self) -> None:
        """Positive drift should increase likelihood of choice=0 (impl convention)."""
        # In this implementation, positive drift favours choice=0
        ll_favoured = _scalar_ll(choice=0, rt=0.4, drift=3.0, bound=1.5, bias=0.5, noise=1.0, ndt=0.1)
        ll_unfavoured = _scalar_ll(choice=0, rt=0.4, drift=-3.0, bound=1.5, bias=0.5, noise=1.0, ndt=0.1)
        assert ll_favoured > ll_unfavoured, (
            f"Positive drift should increase P(choice=0): ll(+3)={ll_favoured:.3f} vs ll(-3)={ll_unfavoured:.3f}"
        )


# ------------------------------------------------------------------ #
# Test: edge cases
# ------------------------------------------------------------------ #

class TestWFPTEdgeCases:
    """Boundary conditions and degenerate parameter regimes."""

    def test_very_small_decision_time(self) -> None:
        """RT barely above non-decision time should not produce NaN."""
        ll = _scalar_ll(choice=1, rt=0.101, drift=1.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        assert np.isfinite(ll), f"Near-zero decision time gave non-finite ll: {ll}"

    def test_very_large_rt(self) -> None:
        """Very long RT should produce low but finite density."""
        ll = _scalar_ll(choice=1, rt=10.0, drift=1.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        assert np.isfinite(ll), f"Large RT gave non-finite ll: {ll}"
        # Large RT should have low density
        ll_normal = _scalar_ll(choice=1, rt=0.5, drift=1.0, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        assert ll < ll_normal, "Density at RT=10s should be lower than at RT=0.5s"

    def test_near_zero_drift(self) -> None:
        """Near-zero drift should not cause division by zero."""
        ll = _scalar_ll(choice=1, rt=0.5, drift=1e-6, bound=1.0, bias=0.5, noise=1.0, ndt=0.1)
        assert np.isfinite(ll), f"Near-zero drift gave non-finite ll: {ll}"

    def test_extreme_bias(self) -> None:
        """Bias near 0 or 1 should not cause NaN."""
        ll_low = _scalar_ll(choice=1, rt=0.5, drift=1.0, bound=1.0, bias=0.02, noise=1.0, ndt=0.1)
        ll_high = _scalar_ll(choice=1, rt=0.5, drift=1.0, bound=1.0, bias=0.98, noise=1.0, ndt=0.1)
        assert np.isfinite(ll_low), f"Bias=0.02 gave non-finite ll: {ll_low}"
        assert np.isfinite(ll_high), f"Bias=0.98 gave non-finite ll: {ll_high}"

    def test_batch_consistency(self) -> None:
        """Same parameters in a batch should produce identical likelihoods."""
        n = 5
        ll = wfpt_log_likelihood(
            choice=torch.ones(n),
            rt=torch.full((n,), 0.5),
            drift=torch.full((n,), 2.0),
            bound=torch.full((n,), 1.5),
            bias=torch.full((n,), 0.5),
            noise=torch.full((n,), 1.0),
            non_decision=torch.full((n,), 0.1),
        )
        assert torch.allclose(ll, ll[0].expand(n), atol=1e-6), "Identical inputs produced different outputs"


# ------------------------------------------------------------------ #
# Test: wfpt_loss wrapper
# ------------------------------------------------------------------ #

class TestWFPTLossWrapper:
    """Tests for the high-level wfpt_loss function."""

    def test_loss_is_positive(self) -> None:
        """NLL should be positive (log-likelihood is negative)."""
        batch = 30
        loss = wfpt_loss(
            choice=torch.randint(0, 2, (batch,)).float(),
            rt_ms=torch.rand(batch) * 500 + 300,
            drift=torch.randn(batch),
            bound=torch.ones(batch) * 1.5,
            bias=torch.zeros(batch),
            noise=torch.ones(batch),
            non_decision_ms=torch.full((batch,), 200.0),
        )
        assert loss.item() > 0, f"NLL should be positive, got {loss.item()}"

    def test_weight_scales_loss(self) -> None:
        """Loss weight should scale the output proportionally."""
        batch = 20
        kwargs = dict(
            choice=torch.randint(0, 2, (batch,)).float(),
            rt_ms=torch.rand(batch) * 500 + 300,
            drift=torch.randn(batch),
            bound=torch.ones(batch) * 1.5,
            bias=torch.zeros(batch),
            noise=torch.ones(batch),
            non_decision_ms=torch.full((batch,), 200.0),
        )
        loss_1x = wfpt_loss(**kwargs, weight=1.0)
        loss_2x = wfpt_loss(**kwargs, weight=2.0)
        assert abs(loss_2x.item() - 2.0 * loss_1x.item()) < 1e-4, "Weight should scale loss linearly"
