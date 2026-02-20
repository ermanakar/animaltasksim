"""PyTorch model for the hybrid DDM + LSTM agent."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridDDMModel(nn.Module):
    """Controller that maps history-aware features to DDM parameters."""

    def __init__(
        self, 
        feature_dim: int, 
        hidden_size: int, 
        device: torch.device, 
        drift_scale: float = 10.0,
        history_bias_scale: float = 0.5,
        history_drift_scale: float = 0.0,
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

        # History bias head (legacy, kept for backward compat with saved models):
        # Reads LSTM hidden state → history bias. Phase 6 experiments showed this
        # cannot learn due to gradient instability. Outputs ~0 at zero init.
        self.history_bias_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.history_bias_head.weight)
        nn.init.zeros_(self.history_bias_head.bias)

        # Separate History Network: bypasses LSTM hidden state entirely.
        # Takes (prev_action, prev_reward) directly → "stay tendency."
        # Models the PFC/basal ganglia history circuit that is anatomically
        # separate from the LIP evidence accumulation circuit (the LSTM/DDM).
        # Zero-initialized output layer → no effect until Phase 7 training.
        self.history_network = nn.Sequential(
            nn.Linear(2, 8),   # (prev_action, prev_reward) → 8 hidden
            nn.ReLU(),
            nn.Linear(8, 1),   # 8 → scalar stay_tendency
        )
        nn.init.zeros_(self.history_network[2].weight)
        nn.init.zeros_(self.history_network[2].bias)
        
        # CRITICAL CALIBRATION: Initialize drift_head bias for realistic drift_gain scale
        # Target: drift_gain ~ 10-15 to match macaque RT dynamics (RT ~ 500-800ms)
        # Formula: drift_gain = softplus(bias) + 1e-3
        # For drift_gain ≈ 12: need bias ≈ 2.5
        nn.init.constant_(self.drift_head.bias, 2.5)
        self.log_noise = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.history_bias_scale = nn.Parameter(torch.tensor(float(history_bias_scale), dtype=torch.float32))
        self.history_drift_scale = nn.Parameter(torch.tensor(float(history_drift_scale), dtype=torch.float32))
        self._min_bound = 0.5
        self._min_non_decision = 150.0  # ms
        
        # Initialize drift_head with stronger weights to enable evidence-dependent RTs
        with torch.no_grad():
            self.drift_head.weight.data *= drift_scale
            self.drift_head.bias.data *= drift_scale

    def init_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state
        
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
        non_decision = F.softplus(self.non_decision_head(h)) + self._min_non_decision
        safe_log_noise = torch.nan_to_num(self.log_noise, nan=0.0, posinf=5.0, neginf=-5.0)
        safe_log_noise = torch.clamp(safe_log_noise, -5.0, 5.0)
        noise = torch.exp(safe_log_noise) + 1e-3

        # Separate history network: (prev_action, prev_reward) → stay_tendency.
        # Bypasses LSTM — models a distinct history processing circuit.
        history_input = x[:, 3:5]  # indices 3=prev_action, 4=prev_reward
        stay_tendency = torch.tanh(self.history_network(history_input))

        outputs = {
            "drift_gain": drift_gain.squeeze(-1),
            "bound": bound.squeeze(-1),
            "bias": bias.squeeze(-1),
            "history_bias": history_bias.squeeze(-1),
            "stay_tendency": stay_tendency.squeeze(-1),
            "non_decision_ms": non_decision.squeeze(-1),
            "noise": noise.squeeze(-1).expand_as(drift_gain.squeeze(-1)),
        }
        return outputs, (h, c)
