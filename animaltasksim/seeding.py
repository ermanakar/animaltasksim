"""Utilities for deterministic seeding across supported frameworks."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

def _seed_torch(seed: int) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _seed_torch(seed)


__all__ = ["seed_everything"]
