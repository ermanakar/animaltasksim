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
            nn.Linear(8, 1),   # 8 → scalar lose_stay_tendency
        )
        nn.init.zeros_(self.win_history_network[2].weight)
        nn.init.zeros_(self.win_history_network[2].bias)
        nn.init.zeros_(self.lose_history_network[2].weight)
        nn.init.zeros_(self.lose_history_network[2].bias)

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

        # Asymmetric history pathways: route through win or lose network
        # based on previous trial outcome. Bypasses LSTM entirely.
        history_input = x[:, 3:5]  # indices 3=prev_action, 4=prev_reward
        prev_reward = x[:, 4:5]
        win_tendency = torch.tanh(self.win_history_network(history_input))
        lose_tendency = torch.tanh(self.lose_history_network(history_input))
        is_win = (prev_reward > 0.5).float()
        stay_tendency = is_win * win_tendency + (1.0 - is_win) * lose_tendency

        outputs = {
            "drift_gain": drift_gain.squeeze(-1),
            "bound": bound.squeeze(-1),
            "bias": bias.squeeze(-1),
            "history_bias": history_bias.squeeze(-1),
            "stay_tendency": stay_tendency.squeeze(-1),
            "win_stay_tendency": win_tendency.squeeze(-1),
            "lose_stay_tendency": lose_tendency.squeeze(-1),
            "non_decision_ms": non_decision.squeeze(-1),
            "noise": noise.squeeze(-1).expand_as(drift_gain.squeeze(-1)),
        }
        return outputs, (h, c)
