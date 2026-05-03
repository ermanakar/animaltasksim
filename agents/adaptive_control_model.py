"""Adaptive control model with value, persistence, and exploration drives."""
from __future__ import annotations

import torch
import torch.nn as nn

from agents.hybrid_model import HybridDDMModel


class AdaptiveControlModel(HybridDDMModel):
    """Hybrid evidence core augmented with adaptive control pressures."""

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        device: torch.device,
        drift_scale: float = 6.0,
        history_bias_scale: float = 2.0,
        history_drift_scale: float = 0.3,
        control_state_enabled: bool = True,
        persistence_enabled: bool = True,
        exploration_enabled: bool = True,
        persistence_learning_rate: float = 0.8,
        switch_learning_rate: float = 0.8,
        reward_learning_rate: float = 0.6,
        control_state_decay: float = 0.7,
        control_state_scale: float = 1.0,
        persistence_bias_scale: float = 1.6,
        exploration_bias_scale: float = 0.8,
        control_residual_limit: float = 0.35,
        control_pressure_limit: float = 0.35,
        control_uncertainty_power: float = 2.0,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            device=device,
            drift_scale=drift_scale,
            history_bias_scale=history_bias_scale,
            history_drift_scale=history_drift_scale,
        )
        self.control_state_enabled = control_state_enabled
        self.persistence_enabled = persistence_enabled
        self.exploration_enabled = exploration_enabled
        self.control_residual_limit = float(control_residual_limit)
        self.control_pressure_limit = float(control_pressure_limit)
        self.control_uncertainty_power = max(1.0, float(control_uncertainty_power))
        self.persistence_head = nn.Sequential(
            nn.Linear(hidden_size + 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.exploration_head = nn.Sequential(
            nn.Linear(hidden_size + 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.arbitration_head = nn.Linear(4, 1)

        self.persistence_learning_logit = nn.Parameter(
            torch.tensor(self._to_logit(persistence_learning_rate), dtype=torch.float32)
        )
        self.switch_learning_logit = nn.Parameter(
            torch.tensor(self._to_logit(switch_learning_rate), dtype=torch.float32)
        )
        self.reward_learning_logit = nn.Parameter(
            torch.tensor(self._to_logit(reward_learning_rate), dtype=torch.float32)
        )
        self.control_state_decay_logit = nn.Parameter(
            torch.tensor(self._to_logit(control_state_decay), dtype=torch.float32)
        )
        self.control_state_scale = nn.Parameter(torch.tensor(control_state_scale, dtype=torch.float32))
        self.persistence_bias_scale = nn.Parameter(torch.tensor(persistence_bias_scale, dtype=torch.float32))
        self.exploration_bias_scale = nn.Parameter(torch.tensor(exploration_bias_scale, dtype=torch.float32))

        nn.init.zeros_(self.persistence_head[-1].weight)
        nn.init.zeros_(self.persistence_head[-1].bias)
        nn.init.zeros_(self.exploration_head[-1].weight)
        nn.init.zeros_(self.exploration_head[-1].bias)
        nn.init.zeros_(self.arbitration_head.weight)
        nn.init.zeros_(self.arbitration_head.bias)

    @staticmethod
    def _to_logit(value: float) -> float:
        clipped = min(max(value, 1e-4), 1.0 - 1e-4)
        return float(torch.logit(torch.tensor(clipped)).item())

    @property
    def effective_persistence_learning_rate(self) -> torch.Tensor:
        """Persistence learning rate constrained to [0, 1]."""
        return torch.sigmoid(self.persistence_learning_logit)

    @property
    def effective_switch_learning_rate(self) -> torch.Tensor:
        """Switch learning rate constrained to [0, 1]."""
        return torch.sigmoid(self.switch_learning_logit)

    @property
    def effective_reward_learning_rate(self) -> torch.Tensor:
        """Reward learning rate constrained to [0, 1]."""
        return torch.sigmoid(self.reward_learning_logit)

    @property
    def effective_control_state_decay(self) -> torch.Tensor:
        """Control-state decay constrained to [0, 1]."""
        return torch.sigmoid(self.control_state_decay_logit)

    def _bounded_signed(self, value: torch.Tensor, limit: float) -> torch.Tensor:
        """Limit a signed residual while preserving gradients around zero."""
        if limit <= 0.0:
            return torch.zeros_like(value)
        return float(limit) * torch.tanh(value / float(limit))

    def _bounded_positive(self, value: torch.Tensor, limit: float) -> torch.Tensor:
        """Limit a non-negative pressure while preserving ordering."""
        if limit <= 0.0:
            return torch.zeros_like(value)
        return float(limit) * torch.tanh(torch.clamp(value, min=0.0) / float(limit))

    def _shape_uncertainty(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Sharpen adaptive-control expression around genuinely ambiguous evidence."""
        return torch.clamp(uncertainty, min=0.0, max=1.0).pow(self.control_uncertainty_power)

    @staticmethod
    def _outcome_valence(reward: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map trial reward to binary success/failure teaching signals."""
        success = (reward > 0.0).to(dtype=reward.dtype) * valid
        failure = (reward <= 0.0).to(dtype=reward.dtype) * valid
        return success, failure

    def init_plastic_state(
        self,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize adaptive control state, eligibility trace, value prediction, and uncertainty gate."""
        control_state = torch.zeros(batch_size, 2, device=self.device)
        eligibility_trace = torch.zeros(batch_size, 2, device=self.device)
        prev_value_prediction = torch.zeros(batch_size, 1, device=self.device)
        prev_uncertainty_gate = torch.zeros(batch_size, 1, device=self.device)
        return control_state, eligibility_trace, prev_value_prediction, prev_uncertainty_gate

    def update_plastic_history(
        self,
        plastic_state: torch.Tensor,
        eligibility_trace: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_value_prediction: torch.Tensor,
        prev_history_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update adaptive control from outcome valence while returning critic delta."""
        action_trace = self._action_one_hot(prev_action)
        valid = (action_trace.sum(dim=1, keepdim=True) > 0).float()
        reward = prev_reward.reshape(-1, 1)
        delta = (reward - prev_value_prediction) * valid
        success_signal, failure_signal = self._outcome_valence(reward, valid)
        if not self.control_state_enabled:
            return torch.zeros_like(plastic_state), torch.zeros_like(eligibility_trace), delta.squeeze(-1)

        alternative_trace = torch.flip(action_trace, dims=[1])
        uncertainty = self._shape_uncertainty(prev_history_gate)

        updated_trace = eligibility_trace * self.effective_control_state_decay + action_trace
        rewarded_stay = self.effective_reward_learning_rate * success_signal * updated_trace
        if self.persistence_enabled:
            uncertain_retry = (
                self.effective_persistence_learning_rate
                * failure_signal
                * uncertainty
                * updated_trace
            )
        else:
            uncertain_retry = torch.zeros_like(updated_trace)
        confident_switch = (
            self.effective_switch_learning_rate
            * failure_signal
            * (1.0 - uncertainty)
        )
        chosen_suppression = confident_switch * updated_trace
        alternative_boost = confident_switch * alternative_trace
        updated_state = (
            self.effective_control_state_decay * plastic_state
            + rewarded_stay
            + uncertain_retry
            - chosen_suppression
            + alternative_boost
        )
        updated_state = torch.clamp(updated_state, min=-3.0, max=3.0)
        return updated_state, updated_trace, delta.squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        plastic_state: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        if plastic_state is None:
            plastic_state = torch.zeros(x.shape[0], 2, device=x.device)
        base_outputs, next_state = super().forward(
            x,
            state,
            plastic_state=torch.zeros_like(plastic_state),
        )
        h, _ = next_state

        if not self.control_state_enabled:
            zero = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            outputs = dict(base_outputs)
            outputs.update(
                {
                    "plastic_stay_tendency": zero,
                    "raw_control_bias": zero,
                    "control_gate": zero,
                    "retry_pressure": zero,
                    "switch_pressure": zero,
                    "win_stay_tendency": zero,
                    "lose_shift_tendency": zero,
                    "lose_stay_tendency": zero,
                    "persistence_pressure": zero,
                    "exploration_pressure": zero,
                    "staleness_signal": zero,
                    "raw_control_residual": zero,
                    "control_residual": zero,
                    "arbitration_adjustment": zero,
                }
            )
            return outputs, next_state

        prev_action = x[:, 3]
        prev_reward = x[:, 4:5]
        uncertainty = self._shape_uncertainty(1.0 - torch.abs(x[:, 0:1]))
        has_prev_action = (prev_action.abs().reshape(-1, 1) > 0.0).float()
        action_trace = self._action_one_hot(prev_action)
        alternative_trace = torch.flip(action_trace, dims=[1])
        selected_control = torch.sum(plastic_state * action_trace, dim=1, keepdim=True)
        alternative_control = torch.sum(plastic_state * alternative_trace, dim=1, keepdim=True)
        raw_control_delta = selected_control - alternative_control
        raw_control_bias = self._bounded_signed(
            self.control_state_scale * raw_control_delta,
            self.control_residual_limit,
        ) * has_prev_action
        control_gate = uncertainty * has_prev_action
        control_bias = raw_control_bias * control_gate
        retry_pressure = self._bounded_positive(raw_control_delta, self.control_pressure_limit) * control_gate
        switch_pressure = self._bounded_positive(-raw_control_delta, self.control_pressure_limit) * control_gate
        staleness_signal = torch.clamp(torch.abs(selected_control), min=0.0, max=1.0) * has_prev_action

        controller_input = torch.cat([h, prev_reward, uncertainty, has_prev_action, staleness_signal], dim=1)
        if self.persistence_enabled:
            persistence_pressure = self._bounded_signed(
                self.persistence_head(controller_input),
                self.control_pressure_limit,
            ) * control_gate
        else:
            persistence_pressure = torch.zeros_like(uncertainty)
        if self.exploration_enabled:
            exploration_pressure = self._bounded_signed(
                self.exploration_head(controller_input),
                self.control_pressure_limit,
            ) * uncertainty * staleness_signal
        else:
            exploration_pressure = torch.zeros_like(uncertainty)

        arbitration_input = torch.cat(
            [
                base_outputs["stay_tendency"].unsqueeze(-1),
                control_bias,
                persistence_pressure,
                exploration_pressure,
            ],
            dim=1,
        )
        arbitration_adjustment = self._bounded_signed(
            self.arbitration_head(arbitration_input),
            self.control_pressure_limit,
        ) * control_gate
        raw_control_residual = (
            control_bias
            + self.persistence_bias_scale * persistence_pressure
            - self.exploration_bias_scale * exploration_pressure
            + arbitration_adjustment
        )
        control_residual = self._bounded_signed(raw_control_residual, self.control_residual_limit)
        stay_tendency = torch.tanh(
            base_outputs["stay_tendency"].unsqueeze(-1) + control_residual
        )

        outputs = dict(base_outputs)
        outputs.update(
            {
                "stay_tendency": stay_tendency.squeeze(-1),
                "plastic_stay_tendency": control_bias.squeeze(-1),
                "raw_control_bias": raw_control_bias.squeeze(-1),
                "control_gate": control_gate.squeeze(-1),
                "retry_pressure": retry_pressure.squeeze(-1),
                "switch_pressure": switch_pressure.squeeze(-1),
                "win_stay_tendency": persistence_pressure.squeeze(-1),
                "lose_shift_tendency": exploration_pressure.squeeze(-1),
                "lose_stay_tendency": (-exploration_pressure).squeeze(-1),
                "persistence_pressure": persistence_pressure.squeeze(-1),
                "exploration_pressure": exploration_pressure.squeeze(-1),
                "staleness_signal": staleness_signal.squeeze(-1),
                "raw_control_residual": raw_control_residual.squeeze(-1),
                "control_residual": control_residual.squeeze(-1),
                "arbitration_adjustment": arbitration_adjustment.squeeze(-1),
            }
        )
        return outputs, next_state


__all__ = ["AdaptiveControlModel"]
