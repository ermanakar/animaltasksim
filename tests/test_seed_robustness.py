"""Check seeded replay and metric accounting, not empirical seed stability.

Cross-seed behavioral spreads are scientific benchmark results, not correctness
invariants. In particular the corrected zero-contrast task exposes substantial
Sticky-Q history variability; passing these tests does not certify robustness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents.sticky_q import StickyGLMHyperParams, StickyGLMTrainingConfig, train_sticky_q
from eval.metrics import load_and_compute
from eval.schema_validator import validate_file

SEEDS = [42, 123, 7, 2024, 9999]
N_EPISODES = 3
TRIALS = 200


def _run_sticky_q(output: Path, seed: int) -> tuple[list[dict], dict]:
    """Return schema-validated trial records and metrics from one seeded run."""
    output.mkdir()
    config = StickyGLMTrainingConfig(
        episodes=N_EPISODES, trials_per_episode=TRIALS, seed=seed, output_dir=output,
        hyperparams=StickyGLMHyperParams(
            learning_rate=0.05, weight_decay=0.0, temperature=1.0, sample_actions=False,
        ),
    )
    train_sticky_q(config)
    log_path = output / "trials.ndjson"
    validate_file(log_path)
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    return rows, load_and_compute(log_path)


def _canonical_trials(rows: list[dict]) -> list[dict]:
    """Normalize random session UUIDs while preserving episode boundaries."""
    sessions: dict[str, int] = {}
    normalized = []
    for row in rows:
        row = dict(row)
        session_id = row["session_id"]
        row["session_id"] = sessions.setdefault(session_id, len(sessions))
        normalized.append(row)
    return normalized


@pytest.mark.parametrize("seed", SEEDS)
def test_seeded_replay_and_history_accounting(tmp_path: Path, seed: int) -> None:
    """Repeat each seed exactly and reconcile history rates with actual trials."""
    rows, metrics = _run_sticky_q(tmp_path / "first", seed)
    replay, _ = _run_sticky_q(tmp_path / "replay", seed)
    assert len(rows) == N_EPISODES * TRIALS
    assert _canonical_trials(rows) == _canonical_trials(replay)
    wins, losses = [], []
    for row in rows:
        previous = row["prev"]
        if previous is None or row["action"] not in {"left", "right"}:
            continue
        if previous["action"] not in {"left", "right"}:
            continue
        if previous["correct"]:
            wins.append(row["action"] == previous["action"])
        else:
            losses.append(row["action"] != previous["action"])
    assert wins and losses
    assert metrics["history"]["win_stay"] == pytest.approx(np.mean(wins))
    assert metrics["history"]["lose_shift"] == pytest.approx(np.mean(losses))
    assert {"psychometric", "history", "quality"} <= metrics.keys()
