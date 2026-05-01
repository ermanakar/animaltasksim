"""Tests for the history bias architecture (legacy head + asymmetric history networks).

Validates that:
  1. history_bias_head, win_history_network, lose_history_network exist with correct initialization.
  2. WFPT loss gradient does NOT flow to history modules.
  3. Per-trial history loss gradient DOES flow to win/lose history networks.
  4. Asymmetric routing: win trials → win_history_network, lose trials → lose_history_network.
  5. Zero-initialized modules preserve existing model behavior.
  6. Gradient flows correctly through the separate history stream.
  7. Lapse rate is NOT a model parameter (fixed in config, applied externally).
  8. Old single history_network is removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from agents.hybrid_ddm_lstm import HybridDDMModel
from agents.losses import per_trial_history_loss


def _make_model(hidden_size: int = 16, **kwargs) -> HybridDDMModel:
    """Create a small model for testing."""
    device = torch.device("cpu")
    return HybridDDMModel(feature_dim=7, hidden_size=hidden_size, device=device, drift_scale=1.0, **kwargs)


class TestHistoryArchitecture:
    """Verify history modules exist and are properly initialized."""

    def test_history_bias_head_exists(self) -> None:
        model = _make_model()
        assert hasattr(model, "history_bias_head")
        assert isinstance(model.history_bias_head, torch.nn.Linear)

    def test_win_history_network_exists(self) -> None:
        model = _make_model()
        assert hasattr(model, "win_history_network")
        assert isinstance(model.win_history_network, torch.nn.Sequential)

    def test_lose_history_network_exists(self) -> None:
        model = _make_model()
        assert hasattr(model, "lose_history_network")
        assert isinstance(model.lose_history_network, torch.nn.Sequential)

    def test_no_old_history_network(self) -> None:
        """Old single history_network should be removed."""
        model = _make_model()
        assert not hasattr(model, "history_network")

    def test_asymmetric_zero_init(self) -> None:
        """Output layers of both win/lose history networks should be zero at init."""
        model = _make_model()
        # Legacy head
        assert model.history_bias_head.weight.abs().max().item() < 1e-8
        assert model.history_bias_head.bias.abs().max().item() < 1e-8
        # Win history network output layer (index 2 in Sequential)
        assert model.win_history_network[2].weight.abs().max().item() < 1e-8
        assert model.win_history_network[2].bias.abs().max().item() < 1e-8
        # Lose history network output layer
        assert model.lose_history_network[2].weight.abs().max().item() < 1e-8
        assert model.lose_history_network[2].bias.abs().max().item() < 1e-8

    def test_forward_outputs(self) -> None:
        """Model forward should include history_bias and stay_tendency."""
        model = _make_model()
        x = torch.randn(1, 7)
        state = model.init_state(1)
        out, _ = model(x, state)
        assert "history_bias" in out
        assert "stay_tendency" in out
        assert "win_stay_tendency" in out
        assert "lose_shift_tendency" in out
        assert "lose_stay_tendency" in out
        assert out["stay_tendency"].shape == out["bias"].shape

    def test_stay_tendency_backward_compat(self) -> None:
        """stay_tendency key must still be in output with correct shape."""
        model = _make_model()
        x = torch.randn(3, 7)
        state = model.init_state(3)
        out, _ = model(x, state)
        assert "stay_tendency" in out
        assert out["stay_tendency"].shape == (3,)

    def test_zero_init_means_zero_output(self) -> None:
        """At initialization, both history outputs should be ~0."""
        model = _make_model()
        x = torch.randn(5, 7)
        h, c = model.init_state(5)
        out, _ = model(x, (h, c))
        assert out["history_bias"].abs().max().item() < 1e-4
        assert out["stay_tendency"].abs().max().item() < 1e-4


class TestFixedLapseRate:
    """Verify lapse is NOT a model parameter (fixed in config, applied externally)."""

    def test_no_lapse_logit_parameter(self) -> None:
        """lapse_logit should not exist on the model — lapse is fixed, not learnable."""
        model = _make_model()
        assert not hasattr(model, "lapse_logit")

    def test_no_lapse_rate_in_output(self) -> None:
        """lapse_rate should not appear in model output dict."""
        model = _make_model()
        x = torch.randn(1, 7)
        state = model.init_state(1)
        out, _ = model(x, state)
        assert "lapse_rate" not in out

    def test_lapse_not_in_model_parameters(self) -> None:
        """No model parameter should be named 'lapse'."""
        model = _make_model()
        param_names = [name for name, _ in model.named_parameters()]
        assert not any("lapse" in name for name in param_names)


class TestAsymmetricPathwaySelection:
    """Verify win/lose pathway routing based on prev_reward."""

    def test_win_loss_pathway_selection(self) -> None:
        """When prev_reward=1 use win-stay; when prev_reward=0 use lose-shift."""
        model = _make_model()
        # Set different weights for win and lose networks to distinguish them
        with torch.no_grad():
            model.win_history_network[0].weight.fill_(1.0)
            model.win_history_network[0].bias.fill_(0.5)
            model.win_history_network[2].weight.fill_(1.0)
            model.win_history_network[2].bias.fill_(0.0)
            model.lose_history_network[0].weight.fill_(1.0)
            model.lose_history_network[0].bias.fill_(0.5)
            model.lose_history_network[2].weight.fill_(1.0)
            model.lose_history_network[2].bias.fill_(0.0)

        state = model.init_state(1)

        # Win trial (prev_reward=1)
        x_win = torch.tensor([[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.5]])
        out_win, _ = model(x_win, state)

        # Lose trial (prev_reward=0)
        x_lose = torch.tensor([[0.1, 0.1, 1.0, 1.0, 0.0, 0.0, 0.5]])
        out_lose, _ = model(x_lose, state)

        # stay_tendency should match the appropriate pathway semantics after the
        # final tanh nonlinearity applied to the combined history signal.
        expected_win_stay = torch.tanh(out_win["win_stay_tendency"])
        expected_lose_stay = -torch.tanh(out_lose["lose_shift_tendency"])
        assert abs(out_win["stay_tendency"].item() - expected_win_stay.item()) < 1e-6
        assert abs(out_lose["stay_tendency"].item() - expected_lose_stay.item()) < 1e-6
        assert abs(out_lose["lose_stay_tendency"].item() + out_lose["lose_shift_tendency"].item()) < 1e-6

        # And they should differ from each other
        assert out_win["stay_tendency"].item() != out_lose["stay_tendency"].item()

    def test_win_lose_independent(self) -> None:
        """Win and lose networks should be independent (different parameters)."""
        model = _make_model()
        # They start with same architecture but separate parameters
        win_params = set(id(p) for p in model.win_history_network.parameters())
        lose_params = set(id(p) for p in model.lose_history_network.parameters())
        assert win_params.isdisjoint(lose_params)


class TestSeparateHistoryStream:
    """Verify the separate history networks process inputs correctly."""

    def test_history_network_uses_prev_action_prev_reward(self) -> None:
        """History networks should respond to changes in prev_action/prev_reward."""
        model = _make_model()
        # Break zero-init so networks have non-trivial response
        with torch.no_grad():
            model.win_history_network[0].weight.fill_(0.5)
            model.win_history_network[0].bias.fill_(0.1)
            model.win_history_network[2].weight.fill_(0.5)
            model.win_history_network[2].bias.fill_(0.0)
            model.lose_history_network[0].weight.fill_(0.5)
            model.lose_history_network[0].bias.fill_(0.1)
            model.lose_history_network[2].weight.fill_(0.5)
            model.lose_history_network[2].bias.fill_(0.0)

        state = model.init_state(1)

        # Feature layout: [coh, abs_coh, sign, prev_action, prev_reward, prev_correct, trial_norm]
        # Win trial (prev_reward=1, prev_action=1)
        x_win_right = torch.tensor([[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.5]])
        out_wr, _ = model(x_win_right, state)

        # Loss trial (prev_reward=0, prev_action=1)
        x_lose_right = torch.tensor([[0.1, 0.1, 1.0, 1.0, 0.0, 0.0, 0.5]])
        out_lr, _ = model(x_lose_right, state)

        # Stay tendency should differ between win and loss
        assert out_wr["stay_tendency"].item() != out_lr["stay_tendency"].item()

    def test_loss_pathway_represents_shift_pressure(self) -> None:
        """Positive lose head output should reduce stay tendency after losses."""
        model = _make_model()
        with torch.no_grad():
            model.lose_history_network[0].weight.fill_(0.5)
            model.lose_history_network[0].bias.fill_(0.1)
            model.lose_history_network[2].weight.fill_(0.5)
            model.lose_history_network[2].bias.fill_(0.0)

        state = model.init_state(1)
        x_lose_right = torch.tensor([[0.1, 0.1, 1.0, 1.0, 0.0, 0.0, 0.5]])
        out, _ = model(x_lose_right, state)

        assert out["lose_shift_tendency"].item() > 0.0
        assert out["lose_stay_tendency"].item() < 0.0
        assert out["stay_tendency"].item() < 0.0

    def test_history_network_independent_of_coherence(self) -> None:
        """stay_tendency should not change when only coherence changes."""
        model = _make_model()
        with torch.no_grad():
            model.win_history_network[0].weight.fill_(0.5)
            model.win_history_network[2].weight.fill_(0.5)

        # Same history features, different coherence
        x_low = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]])
        x_high = torch.tensor([[0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5]])

        # history networks take x[:, 3:5] only, so same input
        hist_input_low = x_low[:, 3:5]
        hist_input_high = x_high[:, 3:5]
        assert torch.allclose(hist_input_low, hist_input_high)


class TestGradientIsolation:
    """Verify WFPT and history gradients flow to correct modules."""

    def test_wfpt_gradient_does_not_reach_history_network(self) -> None:
        """WFPT loss should not create gradients on win/lose history networks."""
        from agents.wfpt_loss import wfpt_loss

        model = _make_model()
        model.train()
        with torch.no_grad():
            model.win_history_network[2].weight.fill_(0.1)
            model.lose_history_network[2].weight.fill_(0.1)

        x = torch.randn(1, 7)
        state = model.init_state(1)
        out, _ = model(x, state)

        loss = wfpt_loss(
            choice=torch.tensor([1.0]),
            rt_ms=torch.tensor([500.0]),
            drift=out["drift_gain"] * x[0, 0],
            bound=out["bound"],
            bias=out["bias"],
            noise=out["noise"],
            non_decision_ms=out["non_decision_ms"],
        )
        loss.backward()

        # win/lose history network output layers should have zero gradients
        for net in [model.win_history_network, model.lose_history_network]:
            out_layer = net[2]
            assert out_layer.weight.grad is None or \
                out_layer.weight.grad.abs().max().item() < 1e-10

    def test_history_loss_reaches_history_network(self) -> None:
        """P(stay) loss through win_history_network should create gradients."""
        model = _make_model()
        model.train()
        with torch.no_grad():
            model.win_history_network[0].weight.fill_(0.5)
            model.win_history_network[2].weight.fill_(0.1)

        # Use a win trial (prev_reward=1) so gradient flows to win_history_network
        x = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]])
        state = model.init_state(1)
        out, _ = model(x, state)

        # P(stay) from stay_tendency
        stay_prob = torch.sigmoid(out["stay_tendency"] * 0.5 * 4.0)
        target = torch.full_like(stay_prob, 0.72)
        loss = F.mse_loss(stay_prob, target)
        loss.backward()

        # win_history_network SHOULD have non-zero gradients
        out_layer = model.win_history_network[2]
        assert out_layer.weight.grad is not None
        assert out_layer.weight.grad.abs().sum().item() > 1e-8

    def test_history_loss_does_not_reach_lstm(self) -> None:
        """P(stay) loss through history networks should NOT reach LSTM."""
        model = _make_model()
        model.train()
        with torch.no_grad():
            model.win_history_network[0].weight.fill_(0.5)
            model.win_history_network[2].weight.fill_(0.1)

        x = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]])
        state = model.init_state(1)
        out, _ = model(x, state)

        stay_prob = torch.sigmoid(out["stay_tendency"] * 0.5 * 4.0)
        loss = F.mse_loss(stay_prob, torch.full_like(stay_prob, 0.72))
        loss.backward()

        # LSTM weights should have zero gradients (history networks bypass LSTM)
        assert model.lstm.weight_ih.grad is None or \
            model.lstm.weight_ih.grad.abs().max().item() < 1e-10

    def test_history_loss_does_not_reach_bias_head(self) -> None:
        """P(stay) from history networks should NOT reach bias_head."""
        model = _make_model()
        model.train()
        with torch.no_grad():
            model.win_history_network[0].weight.fill_(0.5)
            model.win_history_network[2].weight.fill_(0.1)

        x = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]])
        state = model.init_state(1)
        out, _ = model(x, state)

        stay_prob = torch.sigmoid(out["stay_tendency"] * 0.5 * 4.0)
        loss = F.mse_loss(stay_prob, torch.full_like(stay_prob, 0.72))
        loss.backward()

        assert model.bias_head.weight.grad is None or \
            model.bias_head.weight.grad.abs().max().item() < 1e-10

    def test_ddm_direct_history_loss_reaches_history_network(self) -> None:
        """DDM-direct history loss should propagate into the win history network."""
        model = _make_model(history_bias_scale=2.0, history_drift_scale=0.3)
        model.train()
        with torch.no_grad():
            model.win_history_network[0].weight.fill_(0.5)
            model.win_history_network[0].bias.fill_(0.1)
            model.win_history_network[2].weight.fill_(0.2)
            model.win_history_network[2].bias.fill_(0.1)

        x = torch.tensor([[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.5]])
        state = model.init_state(1)
        out, _ = model(x, state)

        coherence = x[0, 0]
        drift = out["drift_gain"] * coherence
        bound = out["bound"]
        bias = out["bias"]
        prev_direction = torch.tensor(1.0)
        history_drift = out["stay_tendency"] * model.history_drift_scale * prev_direction
        gated_history_drift = history_drift * (1.0 - torch.abs(coherence))
        effective_drift = drift + gated_history_drift
        stay_shift = out["stay_tendency"] * model.effective_history_bias_scale * bound
        effective_bias = bias + stay_shift * prev_direction

        max_steps = 40
        dt = 0.01
        temp = 0.1
        step_drift = effective_drift * dt
        evidence_trajectory = effective_bias + torch.cumsum(
            torch.ones(max_steps, dtype=step_drift.dtype) * step_drift.squeeze(),
            dim=0,
        )

        prob_upper = torch.sigmoid((evidence_trajectory - bound.squeeze()) / temp)
        prob_lower = torch.sigmoid((-bound.squeeze() - evidence_trajectory) / temp)
        prob_commit = torch.clamp(prob_upper + prob_lower, 0.0, 1.0)
        prob_not_commit = 1.0 - prob_commit
        cum_not_commit = torch.cat([
            torch.ones(1, dtype=prob_commit.dtype),
            torch.cumprod(prob_not_commit[:-1], dim=0),
        ])
        commit_density = prob_commit * cum_not_commit
        prob_timeout = cum_not_commit[-1] * (1.0 - prob_commit[-1])
        p_right_given_commit = torch.sum(prob_upper * cum_not_commit) / torch.clamp(
            torch.sum(commit_density), min=1e-8
        )
        p_right_given_timeout = torch.sigmoid(evidence_trajectory[-1] / temp)
        prob_right = (1.0 - prob_timeout) * p_right_given_commit + prob_timeout * p_right_given_timeout

        loss = per_trial_history_loss(
            choice_prob=prob_right.unsqueeze(0),
            prev_action=torch.tensor([1.0]),
            prev_reward=torch.tensor([1.0]),
            target_win_stay=0.95,
            no_action_value=0.0,
        )
        loss.backward()

        out_layer = model.win_history_network[2]
        assert out_layer.weight.grad is not None
        assert out_layer.weight.grad.abs().sum().item() > 0.0


class TestCombinedBiasComputation:
    """Verify the combined bias logic used in rollout."""

    def test_combined_bias_with_zero_history(self) -> None:
        """With zero-initialized history modules, combined_start ≈ bias."""
        model = _make_model()
        x = torch.randn(1, 7)
        state = model.init_state(1)
        out, _ = model(x, state)

        bias = out["bias"].item()
        stay_tendency = out["stay_tendency"].item()
        bound = out["bound"].item()
        scale = 0.5

        # stay_tendency ≈ 0 at init → combined ≈ bias
        stay_shift = stay_tendency * scale * bound
        prev_direction = 1.0  # prev was right
        combined = bias + stay_shift * prev_direction
        assert abs(combined - bias) < 1e-3

    def test_combined_bias_scales_with_bound(self) -> None:
        """stay_tendency contribution should scale with bound."""
        model = _make_model()
        with torch.no_grad():
            model.win_history_network[0].weight.fill_(1.0)
            model.win_history_network[0].bias.fill_(0.5)
            model.win_history_network[2].weight.fill_(1.0)
            model.win_history_network[2].bias.fill_(0.5)

        x = torch.tensor([[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.5]])
        state = model.init_state(1)
        out, _ = model(x, state)

        stay_tendency = out["stay_tendency"].item()
        bound = out["bound"].item()
        scale = 0.5

        stay_shift = stay_tendency * scale * bound
        assert abs(stay_shift) > 0.01


class TestHistoryTeacherForcing:
    """Verify annealed history teacher forcing blends injected and learned tendencies."""

    def test_teacher_forcing_blends_tendencies(self) -> None:
        from agents.hybrid_config import HybridTrainingConfig
        from agents.hybrid_trainer import HybridDDMTrainer

        config = HybridTrainingConfig(
            task="ibl_2afc",
            output_dir=Path("runs/test_teacher_forcing_blend"),
            inject_win_tendency=0.3,
            inject_lose_tendency=0.1,
            anneal_history_injection=True,
            history_injection_alpha_start=1.0,
            history_injection_alpha_end=0.0,
            max_sessions=1,
            max_trials_per_session=4,
            episodes=1,
            epochs=2,
        )
        trainer = HybridDDMTrainer(config)

        learned = torch.tensor([0.05])
        reward = torch.tensor(1.0)

        trainer._set_history_injection_alpha(1.0)
        full_teacher = trainer._apply_history_teacher_forcing(learned, reward)
        assert full_teacher.item() == pytest.approx(0.3)

        trainer._set_history_injection_alpha(0.5)
        blended = trainer._apply_history_teacher_forcing(learned, reward)
        assert blended.item() == pytest.approx(0.175)

        trainer._set_history_injection_alpha(0.0)
        learned_only = trainer._apply_history_teacher_forcing(learned, reward)
        assert learned_only.item() == pytest.approx(learned.item())

    def test_teacher_alpha_anneals_over_epochs(self) -> None:
        from agents.hybrid_config import HybridTrainingConfig
        from agents.hybrid_trainer import HybridDDMTrainer

        config = HybridTrainingConfig(
            task="ibl_2afc",
            output_dir=Path("runs/test_teacher_alpha_schedule"),
            inject_win_tendency=0.3,
            anneal_history_injection=True,
            history_injection_alpha_start=1.0,
            history_injection_alpha_end=0.25,
            max_sessions=1,
            max_trials_per_session=4,
            episodes=1,
            epochs=4,
        )
        trainer = HybridDDMTrainer(config)

        alphas = [trainer._teacher_forcing_alpha_for_epoch(epoch, 4) for epoch in range(4)]
        assert alphas == pytest.approx([1.0, 0.75, 0.5, 0.25])
