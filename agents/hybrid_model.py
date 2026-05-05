"""PyTorch model for the hybrid DDM + LSTM agent."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_linear_layer(module: nn.Module) -> None:
    """Zero-initialize a Linear layer after narrowing Sequential members."""
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}")
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class HybridDDMModel(nn.Module):
    """Controller that maps history-aware features to DDM parameters."""

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        device: torch.device,
        drift_scale: float = 10.0,
        history_bias_scale: float = 2.0,
        history_drift_scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        
        # Decompose feature dimensions
        self.stim_dim = 3  # coherence, abs_coh, sign
        self.history_dim = 3  # prev_action, prev_reward, prev_correct
        self.temporal_dim = 1  # trial_index_norm
        
        # Architectural Change: Dedicated History Embedding
        self.history_embed_size = 8
        self.history_embed = nn.Linear(self.history_dim, self.history_embed_size)
        
        lstm_input_dim = self.stim_dim + self.history_embed_size + self.temporal_dim
        self.lstm = nn.LSTMCell(lstm_input_dim, hidden_size)
        
        # DDM parameter heads
        self.drift_head = nn.Linear(hidden_size, 1)
        self.bound_head = nn.Linear(hidden_size, 1)
        self.bias_head = nn.Linear(hidden_size, 1)
        self.non_decision_head = nn.Linear(hidden_size, 1)
        self.critic_head = nn.Linear(hidden_size, 1)

        # History bias head (legacy, kept for backward compat with saved models):
        # Reads LSTM hidden state → history bias. Phase 6 experiments showed this
        # cannot learn due to gradient instability. Outputs ~0 at zero init.
        self.history_bias_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.history_bias_head.weight)
        nn.init.zeros_(self.history_bias_head.bias)

        # Asymmetric History Networks: separate win/lose pathways model the
        # dopaminergic asymmetry between reward and punishment processing.
        # Animals show win-stay >> lose-shift (e.g., IBL mouse: 0.724 vs 0.427).
        # A single network produces symmetric effects; splitting allows
        # independent learning of win-stay and lose-shift tendencies.
        # Both bypass LSTM — models PFC/basal ganglia history circuits.
        # Zero-initialized output layers → no effect until history training.
        self.win_history_network = nn.Sequential(
            nn.Linear(2, 8),   # (prev_action, prev_reward) → 8 hidden
            nn.ReLU(),
            nn.Linear(8, 1),   # 8 → scalar win_stay_tendency
        )
        self.lose_history_network = nn.Sequential(
            nn.Linear(2, 8),   # (prev_action, prev_reward) → 8 hidden
            nn.ReLU(),
            nn.Linear(8, 1),   # 8 → scalar lose_shift_tendency
        )
        _zero_linear_layer(self.win_history_network[2])
        _zero_linear_layer(self.lose_history_network[2])
        nn.init.zeros_(self.critic_head.weight)
        nn.init.zeros_(self.critic_head.bias)

        # Plastic history subsystem: action-specific fast weights updated online
        # by a reward-prediction-error signal and an eligibility trace.
        # Use separate coefficients for appetitive and aversive learning so the
        # model can express win-stay and lose-shift asymmetrically.
        self.positive_history_plasticity = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        self.negative_history_plasticity = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.counterfactual_switch_boost = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.history_trace_decay = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.history_state_decay = nn.Parameter(torch.tensor(1.4, dtype=torch.float32))
        self.plastic_history_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # CRITICAL CALIBRATION: Initialize drift_head bias for realistic drift_gain scale
        # Target: drift_gain ~ 10-15 to match macaque RT dynamics (RT ~ 500-800ms)
        # Formula: drift_gain = softplus(bias) + 1e-3
        # For drift_gain ≈ 12: need bias ≈ 2.5
        nn.init.constant_(self.drift_head.bias, 2.5)
        self.log_noise = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.history_bias_scale = nn.Parameter(torch.tensor(float(history_bias_scale), dtype=torch.float32))
        self.history_drift_scale = nn.Parameter(torch.tensor(float(history_drift_scale), dtype=torch.float32))
        # Minimum floor for history scales — prevents optimizer from shrinking
        # them to near-zero, which collapses the sigmoid range and kills
        # history effects (e.g., scale=0.5 → sigmoid max 0.622, can't reach
        # WS target 0.724). Floor of 1.0 gives sigmoid range [0.27, 0.73].
        self._min_history_bias_scale = 1.0
        self._min_bound = 0.5
        self._min_non_decision = 150.0  # ms
        
        # Initialize drift_head with stronger weights to enable evidence-dependent RTs
        with torch.no_grad():
            self.drift_head.weight.data *= drift_scale
            self.drift_head.bias.data *= drift_scale

    @property
    def effective_history_bias_scale(self) -> torch.Tensor:
        """History bias scale with floor clamp to prevent sigmoid collapse."""
        return torch.clamp(self.history_bias_scale, min=self._min_history_bias_scale)

    def init_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return h, c

    def init_plastic_state(
        self,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize fast history weights, eligibility trace, critic state, and gate."""
        plastic_state = torch.zeros(batch_size, 2, device=self.device)
        eligibility_trace = torch.zeros(batch_size, 2, device=self.device)
        prev_value_prediction = torch.zeros(batch_size, 1, device=self.device)
        prev_history_gate = torch.zeros(batch_size, 1, device=self.device)
        return plastic_state, eligibility_trace, prev_value_prediction, prev_history_gate

    @property
    def effective_positive_history_plasticity(self) -> torch.Tensor:
        """Reward-driven plasticity rate constrained to a stable range."""
        return torch.sigmoid(self.positive_history_plasticity)

    @property
    def effective_negative_history_plasticity(self) -> torch.Tensor:
        """Punishment-driven plasticity rate constrained to a stable range."""
        return torch.sigmoid(self.negative_history_plasticity)

    @property
    def effective_counterfactual_switch_boost(self) -> torch.Tensor:
        """Strengthen the alternative action after negative outcomes."""
        return torch.sigmoid(self.counterfactual_switch_boost)

    @property
    def effective_history_trace_decay(self) -> torch.Tensor:
        """Eligibility-trace decay constrained to [0, 1]."""
        return torch.sigmoid(self.history_trace_decay)

    @property
    def effective_history_state_decay(self) -> torch.Tensor:
        """Fast-weight persistence constrained to [0, 1]."""
        return torch.sigmoid(self.history_state_decay)

    @property
    def effective_plastic_history_scale(self) -> torch.Tensor:
        """Scale for converting fast weights into a stay tendency."""
        return 2.0 * torch.sigmoid(self.plastic_history_scale)

    @staticmethod
    def _action_one_hot(prev_action: torch.Tensor) -> torch.Tensor:
        """Map previous action {-1, 0, 1} to [left, right] one-hot encoding."""
        prev_action = prev_action.reshape(-1, 1)
        left = (prev_action < 0).float()
        right = (prev_action > 0).float()
        return torch.cat([left, right], dim=1)

    def update_plastic_history(
        self,
        plastic_state: torch.Tensor,
        eligibility_trace: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_value_prediction: torch.Tensor,
        prev_history_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a local dopamine-like update to the fast history weights."""
        action_trace = self._action_one_hot(prev_action)
        alternative_trace = torch.flip(action_trace, dims=[1])
        valid = (action_trace.sum(dim=1, keepdim=True) > 0).float()
        reward = prev_reward.reshape(-1, 1)
        delta = (reward - prev_value_prediction) * valid
        positive_delta = torch.clamp(delta, min=0.0)
        negative_delta = torch.clamp(-delta, min=0.0)
        updated_trace = (
            self.effective_history_trace_decay * eligibility_trace
            + prev_history_gate * action_trace
        )
        positive_update = self.effective_positive_history_plasticity * positive_delta * updated_trace
        negative_update = self.effective_negative_history_plasticity * negative_delta * updated_trace
        switch_update = (
            self.effective_counterfactual_switch_boost
            * self.effective_negative_history_plasticity
            * negative_delta
            * alternative_trace
        )
        updated_state = (
            self.effective_history_state_decay * plastic_state
            + positive_update
            - negative_update
            + switch_update
        )
        updated_state = torch.clamp(updated_state, min=-3.0, max=3.0)
        return updated_state, updated_trace, delta.squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        plastic_state: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state
        if plastic_state is None:
            plastic_state = torch.zeros(x.shape[0], 2, device=x.device)
        
        # Decompose input and process history
        stim_features = x[:, :self.stim_dim]
        history_features = x[:, self.stim_dim:self.stim_dim + self.history_dim]
        temporal_features = x[:, self.stim_dim + self.history_dim:]
        
        history_embedding = F.relu(self.history_embed(history_features))
        
        # Concatenate features for LSTM
        lstm_input = torch.cat([stim_features, history_embedding, temporal_features], dim=1)
        
        h, c = self.lstm(lstm_input, (h_prev, c_prev))
        drift_gain = F.softplus(self.drift_head(h)) + 1e-3
        bound = F.softplus(self.bound_head(h)) + self._min_bound
        bias = torch.tanh(self.bias_head(h))
        history_bias = torch.tanh(self.history_bias_head(h))
        critic_value = torch.sigmoid(self.critic_head(h))
        non_decision = F.softplus(self.non_decision_head(h)) + self._min_non_decision
        safe_log_noise = torch.nan_to_num(self.log_noise, nan=0.0, posinf=5.0, neginf=-5.0)
        safe_log_noise = torch.clamp(safe_log_noise, -5.0, 5.0)
        noise = torch.exp(safe_log_noise) + 1e-3

        # Asymmetric history pathways: route through win or lose network
        # based on previous trial outcome. The reward pathway emits explicit
        # stay pressure; the loss pathway emits explicit shift pressure.
        # This matches the behavioural asymmetry more directly than asking the
        # model to discover that "negative stay" should mean "switch".
        history_input = x[:, 3:5]  # indices 3=prev_action, 4=prev_reward
        prev_reward = x[:, 4:5]
        win_tendency = torch.tanh(self.win_history_network(history_input))
        lose_shift_tendency = torch.tanh(self.lose_history_network(history_input))
        is_win = (prev_reward > 0.5).float()
        learned_stay_tendency = is_win * win_tendency - (1.0 - is_win) * lose_shift_tendency

        action_trace = self._action_one_hot(x[:, 3])
        has_prev_action = (action_trace.sum(dim=1, keepdim=True) > 0).float()
        alternative_trace = torch.flip(action_trace, dims=[1])
        selected_fast_weight = torch.sum(plastic_state * action_trace, dim=1, keepdim=True)
        alternative_fast_weight = torch.sum(plastic_state * alternative_trace, dim=1, keepdim=True)
        plastic_tendency = torch.tanh(
            self.effective_plastic_history_scale
            * (selected_fast_weight - alternative_fast_weight)
        ) * has_prev_action
        stay_tendency = torch.tanh(learned_stay_tendency + plastic_tendency)

        outputs = {
            "drift_gain": drift_gain.squeeze(-1),
            "bound": bound.squeeze(-1),
            "bias": bias.squeeze(-1),
            "history_bias": history_bias.squeeze(-1),
            "stay_tendency": stay_tendency.squeeze(-1),
            "plastic_stay_tendency": plastic_tendency.squeeze(-1),
            "win_stay_tendency": win_tendency.squeeze(-1),
            "lose_shift_tendency": lose_shift_tendency.squeeze(-1),
            "lose_stay_tendency": (-lose_shift_tendency).squeeze(-1),
            "critic_value": critic_value.squeeze(-1),
            "non_decision_ms": non_decision.squeeze(-1),
            "noise": noise.squeeze(-1).expand_as(drift_gain.squeeze(-1)),
        }
        return outputs, (h, c)
