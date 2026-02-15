"""
Recurrent Drift-Diffusion Model (R-DDM) package.

Exposes configuration dataclasses, dataset utilities, model definition, and
training entrypoints for the history-aware DDM agent tailored to IBL 2AFC.
"""

from __future__ import annotations

from .config import RDDMConfig, RDDMTrainingSchedule
from .dataset import IBLRDDMDataset, RDMRDDMDataset, rddm_collate_sessions
from .model import RDDMModel, RDDMOutputs
from .trainer import RDDMTrainer, RDDMTrainingState

__all__ = [
    "RDDMConfig",
    "RDDMTrainingSchedule",
    "IBLRDDMDataset",
    "RDMRDDMDataset",
    "rddm_collate_sessions",
    "RDDMModel",
    "RDDMOutputs",
    "RDDMTrainer",
    "RDDMTrainingState",
]
