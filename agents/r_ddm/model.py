"""Neural architecture for the Recurrent Drift-Diffusion Model (R-DDM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import RDDMConfig


@dataclass(slots=True)
class RDDMOutputs:
    """Container for per-trial parameter predictions."""

    drift: torch.Tensor
    boundary: torch.Tensor
    non_decision: torch.Tensor
    bias: torch.Tensor
    noise: torch.Tensor
    choice_prob: torch.Tensor


class RDDMModel(nn.Module):
    """GRU-based recurrent model that predicts DDM parameters per trial."""

    def __init__(self, config: RDDMConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_features

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.param_head = nn.Linear(config.hidden_size, 5)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> RDDMOutputs:
        """
        Predict DDM parameters for every trial in the batch.

        Args:
            features: Tensor of shape [batch, max_len, input_features].
            lengths: Sequence lengths for each batch element.
        """
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)

        params = self.param_head(unpacked)
        return self._transform_params(params)

    def step(self, feature: torch.Tensor, hidden: torch.Tensor | None = None) -> Tuple[RDDMOutputs, torch.Tensor]:
        """
        Single-trial inference used for rollouts.

        Args:
            feature: Tensor with shape [1, features].
            hidden: Optional hidden state from previous call.
        """
        feature = feature.unsqueeze(0)
        if feature.dim() == 2:
            feature = feature.unsqueeze(0)  # [1, 1, F]
        if hidden is not None:
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
        else:
            hidden = torch.zeros(
                self.config.num_layers,
                1,
                self.config.hidden_size,
                device=feature.device,
                dtype=feature.dtype,
            )
        output, hidden_next = self.gru(feature, hidden)
        params = self.param_head(output)  # [1, 1, 5]
        outputs = self._transform_params(params.squeeze(0))
        return outputs, hidden_next

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """Return a zero-initialised hidden state."""
        return torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size, device=self.param_head.weight.device)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _transform_params(self, raw_params: torch.Tensor) -> RDDMOutputs:
        """
        Convert raw network outputs into physically interpretable parameters.

        Returns tensors shaped [batch, seq_len] matching the unpacked GRU output.
        """
        drift_raw, boundary_raw, ndt_raw, bias_raw, noise_raw = torch.unbind(raw_params, dim=-1)

        drift = torch.tanh(drift_raw) * self.config.max_drift
        boundary = F.softplus(boundary_raw) + self.config.min_boundary
        non_decision = F.softplus(ndt_raw) + self.config.min_non_decision
        noise = F.softplus(noise_raw) + self.config.min_noise
        bias_fraction = torch.sigmoid(bias_raw)  # in (0, 1)

        choice_prob = self._choice_probability(drift, boundary, bias_fraction, noise)

        return RDDMOutputs(
            drift=drift,
            boundary=boundary,
            non_decision=non_decision,
            bias=bias_fraction,
            noise=noise,
            choice_prob=choice_prob,
        )

    def _choice_probability(
        self,
        drift: torch.Tensor,
        boundary: torch.Tensor,
        bias_fraction: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Probability of hitting the upper boundary (choosing right)."""
        bias = torch.clamp(bias_fraction, 1e-3, 1.0 - 1e-3) * boundary
        scaled_drift = drift / torch.clamp(noise, min=1e-3)

        numerator = 1.0 - torch.exp(-2.0 * scaled_drift * bias)
        denominator = 1.0 - torch.exp(-2.0 * scaled_drift * boundary)

        # Handle near-zero drift case with limit bias/boundary
        close_to_zero = torch.isclose(scaled_drift, torch.zeros(1, device=drift.device), atol=1e-4)
        ratio = numerator / torch.clamp(denominator, min=1e-6)
        ratio = torch.where(close_to_zero, bias_fraction, ratio)
        return torch.clamp(ratio, 1e-4, 1.0 - 1e-4)