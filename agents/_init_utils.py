"""Initialization helpers shared by agent models."""
from __future__ import annotations

import torch.nn as nn


def zero_linear_layer(module: nn.Module) -> None:
    """Zero-initialize a Linear layer after narrowing generic modules."""
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}")
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


__all__ = ["zero_linear_layer"]
