"""Hybrid DDM + LSTM agent designed to mimic animal behaviour on RDM.
Refactored into smaller modules. This file serves as a facade.
"""
from agents.losses import LossWeights
from agents.curriculum import CurriculumPhase, CurriculumConfig
from agents.hybrid_config import HybridDDMPaths, HybridTrainingConfig, SessionBatch
from agents.hybrid_model import HybridDDMModel
from agents.hybrid_trainer import HybridDDMTrainer, train_hybrid, train_hybrid_curriculum

__all__ = [
    "LossWeights",
    "CurriculumPhase",
    "CurriculumConfig",
    "HybridTrainingConfig",
    "HybridDDMPaths",
    "SessionBatch",
    "HybridDDMModel",
    "HybridDDMTrainer",
    "train_hybrid",
    "train_hybrid_curriculum",
]
