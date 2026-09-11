"""Regression checks for differentiable objectives and session continuity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from agents.hybrid_config import HybridTrainingConfig
from agents.hybrid_trainer import HybridDDMTrainer
from agents.losses import LossWeights


def _rows() -> pd.DataFrame:
    actions = ["right", "right", "left", "right", "left", "left"]
    rewards = [1.0, -0.1, 1.0, -0.1, 1.0, -0.1]
    return pd.DataFrame([
        dict(task="ibl_2afc", session_id="mouse-a", trial_index=i,
             stimulus_contrast=0.0, action=action, correct=rewards[i] > 0,
             reward=rewards[i], rt_ms=50.0 + 10 * i,
             prev_action=actions[i - 1] if i else None,
             prev_reward=rewards[i - 1] if i else None,
             prev_correct=rewards[i - 1] > 0 if i else None)
        for i, action in enumerate(actions)
    ])


def _trainer(monkeypatch: pytest.MonkeyPatch, weights: LossWeights) -> HybridDDMTrainer:
    monkeypatch.setattr("agents.hybrid_trainer.load_trials", lambda _: _rows())
    return HybridDDMTrainer(HybridTrainingConfig(
        task="ibl_2afc", epochs=1, hidden_size=8, max_trials_per_session=3,
        max_commit_steps=12, min_commit_steps=2, loss_weights=weights,
    ))


@pytest.mark.parametrize("objective", ["history", "history_supervision"])
def test_history_objective_updates_model(monkeypatch: pytest.MonkeyPatch, objective: str) -> None:
    weights = LossWeights(choice=0.0, rt=0.0, history=0.0)
    setattr(weights, objective, 1.0)
    trainer = _trainer(monkeypatch, weights)
    before = trainer.model.bias_head.weight.detach().clone()
    metrics = trainer.train()
    assert not torch.equal(before, trainer.model.bias_head.weight)
    assert all(np.isfinite(values).all() for values in metrics.values())


def test_chunks_keep_session_state_and_time_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer(monkeypatch, LossWeights(choice=1.0, rt=0.0, history=0.0))
    assert [b.starts_session for b in trainer.sessions] == [True, False]
    np.testing.assert_allclose(np.concatenate([b.features[:, 6] for b in trainer.sessions]), np.arange(6) / 6)
    calls = []
    original = trainer.model.init_state

    def counted(*args, **kwargs):
        if torch.is_grad_enabled():
            calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer.model, "init_state", counted)
    trainer.train()
    assert len(calls) == 1


def test_history_targets_do_not_bridge_omissions() -> None:
    rows = _rows().iloc[:3].copy()
    rows.loc[1, "action"] = "no_op"
    assert HybridDDMTrainer._session_history_stats(rows) == (0.0, 0.0)


def test_rt_targets_come_from_training_data(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer(monkeypatch, LossWeights())
    target, variance = trainer._compute_rt_targets(
        np.array([100.0, 300.0, 0.0]), np.array([1.0, 1.0, 0.0]),
        np.array([[0.0], [0.0], [1.0]]),
    )
    np.testing.assert_allclose(target, [200.0, 200.0, 200.0])
    np.testing.assert_allclose(variance, [10000.0, 10000.0, 10000.0])
