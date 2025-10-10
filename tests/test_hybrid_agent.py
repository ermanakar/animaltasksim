from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from agents.hybrid_ddm_lstm import HybridTrainingConfig, LossWeights, train_hybrid


def test_hybrid_training_smoke(tmp_path: Path) -> None:
    config = HybridTrainingConfig(
        reference_log=Path("data/macaque/reference.ndjson"),
        output_dir=tmp_path / "hybrid_run",
        trials_per_episode=40,
        episodes=1,
        epochs=1,
        hidden_size=16,
        learning_rate=5e-4,
        loss_weights=LossWeights(choice=1.0, rt=0.1, history=0.0),
        max_sessions=1,
        max_trials_per_session=30,
        min_commit_steps=5,
        max_commit_steps=80,
    )

    result = train_hybrid(config)

    assert "training_metrics" in result
    assert "rollout_stats" in result
    paths = result["paths"]
    log_path = Path(paths["log"])
    assert log_path.exists()

    # Ensure at least one log entry and JSON decodes cleanly
    with log_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    assert lines, "expected trials.ndjson to contain records"
    first = json.loads(lines[0])
    assert first["task"] == "rdm"

    metrics_path = Path(paths["metrics"])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "training" in payload
    assert np.isfinite(payload["rollout"]["mean_reward"])
