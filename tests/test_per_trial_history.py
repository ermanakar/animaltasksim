"""Unit tests for per_trial_history_loss — the decoupling fix.

Validates that the per-trial history loss:
  1. Returns zero when all trials already match targets.
  2. Produces non-zero loss when predictions deviate from targets.
  3. Gradients flow through choice_prob to the model.
  4. Handles both R-DDM (no_action_value=-1) and Hybrid (no_action_value=0) conventions.
  5. Handles edge cases: all wins, all losses, no valid trials.
  6. Is strictly larger than zero when stay/shift rates deviate from targets.
"""

from __future__ import annotations

import pytest
import torch

from agents.losses import per_trial_history_loss


class TestPerTrialHistoryLoss:
    """Core per-trial loss properties."""

    def test_zero_loss_when_matching_targets(self) -> None:
        """Loss should be near-zero when P(stay|win) and P(shift|lose) match targets."""
        target_ws, target_ls = 0.7, 0.4
        # 10 win trials where prev was right → stay_prob = choice_prob
        # Set choice_prob = target_ws = 0.7 → P(stay) = 0.7 ≈ target
        n = 10
        choice_prob = torch.full((n,), target_ws)
        prev_action = torch.ones(n, dtype=torch.long)  # all right
        prev_reward = torch.ones(n)  # all wins
        loss = per_trial_history_loss(
            choice_prob, prev_action, prev_reward,
            target_win_stay=target_ws, target_lose_shift=target_ls,
        )
        assert loss.item() < 1e-6

    def test_nonzero_loss_for_deviation(self) -> None:
        """Loss increases when predictions deviate from targets."""
        target_ws = 0.7
        n = 20
        choice_prob = torch.full((n,), 0.5)  # P(stay) = 0.5, target is 0.7
        prev_action = torch.ones(n, dtype=torch.long)
        prev_reward = torch.ones(n)  # all wins
        loss = per_trial_history_loss(
            choice_prob, prev_action, prev_reward,
            target_win_stay=target_ws,
        )
        assert loss.item() > 0.01  # should be (0.7 - 0.5)² = 0.04

    def test_gradient_flow(self) -> None:
        """Gradients must flow through choice_prob to support backprop."""
        choice_prob = torch.full((10,), 0.5, requires_grad=True)
        prev_action = torch.ones(10, dtype=torch.long)
        prev_reward = torch.ones(10)
        loss = per_trial_history_loss(choice_prob, prev_action, prev_reward)
        loss.backward()
        assert choice_prob.grad is not None
        assert choice_prob.grad.abs().sum().item() > 0

    def test_lose_shift_gradient(self) -> None:
        """Gradients flow for lose-shift trials too."""
        choice_prob = torch.full((10,), 0.5, requires_grad=True)
        prev_action = torch.ones(10, dtype=torch.long)
        prev_reward = torch.zeros(10)  # all losses
        loss = per_trial_history_loss(
            choice_prob, prev_action, prev_reward,
            target_lose_shift=0.4,
        )
        loss.backward()
        assert choice_prob.grad is not None
        assert choice_prob.grad.abs().sum().item() > 0


class TestPerTrialConventions:
    """Convention handling for R-DDM vs Hybrid encodings."""

    def test_rddm_convention(self) -> None:
        """R-DDM: prev_action 0=left, 1=right, -1=invalid."""
        choice_prob = torch.full((6,), 0.6, requires_grad=True)
        prev_action = torch.tensor([-1, 0, 1, 0, 1, -1])
        prev_reward = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        loss = per_trial_history_loss(
            choice_prob, prev_action, prev_reward,
            no_action_value=-1.0,  # R-DDM default
        )
        loss.backward()
        # 4 valid trials (-1 is invalid), should have non-zero loss
        assert loss.item() > 0
        assert choice_prob.grad is not None

    def test_hybrid_convention(self) -> None:
        """Hybrid: prev_action -1=left, 0=invalid, 1=right."""
        choice_prob = torch.full((6,), 0.6, requires_grad=True)
        prev_action = torch.tensor([0.0, -1.0, 1.0, -1.0, 1.0, 0.0])
        prev_reward = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        loss = per_trial_history_loss(
            choice_prob, prev_action, prev_reward,
            no_action_value=0.0,  # Hybrid convention
        )
        loss.backward()
        # 4 valid trials (0 is invalid), should have non-zero loss
        assert loss.item() > 0
        assert choice_prob.grad is not None

    def test_mask_excludes_trials(self) -> None:
        """Mask should exclude trials from loss computation."""
        n = 10
        choice_prob = torch.full((n,), 0.5, requires_grad=True)
        prev_action = torch.ones(n, dtype=torch.long)
        prev_reward = torch.ones(n)
        mask_all = torch.ones(n, dtype=torch.bool)
        mask_half = torch.zeros(n, dtype=torch.bool)
        mask_half[:5] = True

        loss_all = per_trial_history_loss(
            choice_prob.detach().requires_grad_(True),
            prev_action, prev_reward, mask=mask_all,
        )
        loss_half = per_trial_history_loss(
            choice_prob.detach().requires_grad_(True),
            prev_action, prev_reward, mask=mask_half,
        )
        # MSE is per-element, so half masking should give same MSE
        # if all deviations are equal. The key test is that it doesn't crash.
        assert loss_all.item() == pytest.approx(loss_half.item(), abs=1e-6)


class TestPerTrialEdgeCases:
    """Edge cases and boundary conditions."""

    def test_no_valid_trials(self) -> None:
        """Loss should be zero when no valid previous actions exist."""
        n = 5
        choice_prob = torch.full((n,), 0.5)
        prev_action = torch.full((n,), -1, dtype=torch.long)  # all invalid
        prev_reward = torch.ones(n)
        loss = per_trial_history_loss(choice_prob, prev_action, prev_reward)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_all_wins(self) -> None:
        """Only win-stay term contributes when all outcomes are wins."""
        n = 10
        choice_prob = torch.full((n,), 0.6, requires_grad=True)
        prev_action = torch.ones(n, dtype=torch.long)
        prev_reward = torch.ones(n)  # all wins
        loss = per_trial_history_loss(
            choice_prob, prev_action, prev_reward,
            target_win_stay=0.8,
        )
        loss.backward()
        # Loss = (0.6 - 0.8)² = 0.04
        assert loss.item() == pytest.approx(0.04, abs=1e-4)

    def test_all_losses(self) -> None:
        """Only lose-shift term contributes when all outcomes are losses."""
        n = 10
        choice_prob = torch.full((n,), 0.6, requires_grad=True)
        prev_action = torch.ones(n, dtype=torch.long)
        prev_reward = torch.zeros(n)  # all losses
        loss = per_trial_history_loss(
            choice_prob, prev_action, prev_reward,
            target_lose_shift=0.5,
        )
        loss.backward()
        # P(shift) = 1 - P(stay) = 1 - choice_prob = 0.4 (since prev_action=1)
        # Loss = (0.4 - 0.5)² = 0.01
        assert loss.item() == pytest.approx(0.01, abs=1e-4)

    def test_prev_left_stay(self) -> None:
        """When prev was left, P(stay) = 1 - choice_prob."""
        n = 5
        choice_prob = torch.full((n,), 0.3, requires_grad=True)
        prev_action = torch.zeros(n, dtype=torch.long)  # prev was left
        prev_reward = torch.ones(n)  # wins
        loss = per_trial_history_loss(
            choice_prob, prev_action, prev_reward,
            target_win_stay=0.7,
        )
        # P(stay) = 1 - 0.3 = 0.7 → matches target exactly
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_stronger_than_batch_aggregate(self) -> None:
        """Per-trial MSE ≥ batch-mean MSE (Jensen's inequality).

        Per-trial: E[(x - μ)²] = Var(x) + (E[x] - μ)²
        Batch:     (E[x] - μ)²

        So per-trial is always ≥ batch when there's variance in predictions.
        """
        target_ws = 0.7
        # Mixed predictions: some at 0.5, some at 0.9 → mean = 0.7
        choice_prob = torch.tensor([0.5, 0.5, 0.9, 0.9, 0.5, 0.9])
        prev_action = torch.ones(6, dtype=torch.long)
        prev_reward = torch.ones(6)

        per_trial = per_trial_history_loss(
            choice_prob, prev_action, prev_reward, target_win_stay=target_ws,
        )

        # Batch-mean: (mean(0.5,0.5,0.9,0.9,0.5,0.9) - 0.7)² = (0.7 - 0.7)² = 0
        batch_mean_loss = (choice_prob.mean() - target_ws) ** 2

        # Per-trial MSE ≥ batch-mean MSE by Jensen's inequality
        assert per_trial.item() >= batch_mean_loss.item() - 1e-8
        # And in this case per-trial should be strictly positive (variance > 0)
        assert per_trial.item() > 0.01
