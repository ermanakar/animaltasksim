"""Seed robustness tests: verify metrics are stable across seeds.

Runs the same agent configuration with different seeds and checks that
key metrics don't vary wildly. This catches agents whose behaviour is
an artefact of a lucky seed rather than genuine learning.

These tests are slow (~30s each) so they're marked with @pytest.mark.slow.
Run with: pytest tests/test_seed_robustness.py -v --timeout=120
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from agents.sticky_q import StickyGLMHyperParams, StickyGLMTrainingConfig, train_sticky_q
from eval.metrics import load_and_compute


SEEDS = [42, 123, 7, 2024, 9999]
N_EPISODES = 3
TRIALS = 200


def _run_sticky_q(tmp_path: Path, seed: int) -> dict:
    """Train Sticky-Q with a given seed and return metrics."""
    out = tmp_path / f"seed_{seed}"
    out.mkdir()
    config = StickyGLMTrainingConfig(
        episodes=N_EPISODES,
        trials_per_episode=TRIALS,
        seed=seed,
        output_dir=out,
        hyperparams=StickyGLMHyperParams(
            learning_rate=0.05,
            weight_decay=0.0,
            temperature=1.0,
            sample_actions=False,
        ),
    )
    train_sticky_q(config)
    log_path = out / "trials.ndjson"
    assert log_path.exists(), f"No log at {log_path}"
    return load_and_compute(log_path)


class TestSeedRobustness:
    """Verify that key metrics are stable across different random seeds."""

    @pytest.fixture(scope="class")
    def multi_seed_metrics(self, tmp_path_factory) -> list[dict]:
        """Run the same agent with 5 seeds and collect metrics."""
        tmp = tmp_path_factory.mktemp("seed_robustness")
        results = []
        for seed in SEEDS:
            metrics = _run_sticky_q(tmp, seed)
            results.append(metrics)
        return results

    def test_psychometric_slope_stable(self, multi_seed_metrics: list[dict]) -> None:
        """Psychometric slope should be consistent across seeds."""
        slopes = [
            m["psychometric"]["slope"]
            for m in multi_seed_metrics
            if m.get("psychometric", {}).get("slope") is not None
            and np.isfinite(m["psychometric"]["slope"])
        ]
        assert len(slopes) >= 3, f"Too few valid slopes: {len(slopes)}"
        cv = statistics.stdev(slopes) / max(abs(statistics.mean(slopes)), 1e-6)
        assert cv < 1.0, (
            f"Psychometric slope is too variable across seeds: "
            f"CV={cv:.2f}, values={slopes}"
        )

    def test_bias_stable(self, multi_seed_metrics: list[dict]) -> None:
        """Bias should not flip sign wildly across seeds."""
        biases = [
            m["psychometric"]["bias"]
            for m in multi_seed_metrics
            if m.get("psychometric", {}).get("bias") is not None
            and np.isfinite(m["psychometric"]["bias"])
        ]
        assert len(biases) >= 3
        spread = max(biases) - min(biases)
        assert spread < 60.0, (
            f"Bias spread too wide: {spread:.1f}, values={biases}"
        )

    def test_history_metrics_stable(self, multi_seed_metrics: list[dict]) -> None:
        """Win-stay and lose-shift should be qualitatively consistent."""
        win_stays = [
            m["history"]["win_stay"]
            for m in multi_seed_metrics
            if m.get("history", {}).get("win_stay") is not None
            and np.isfinite(m["history"]["win_stay"])
        ]
        lose_shifts = [
            m["history"]["lose_shift"]
            for m in multi_seed_metrics
            if m.get("history", {}).get("lose_shift") is not None
            and np.isfinite(m["history"]["lose_shift"])
        ]
        assert len(win_stays) >= 3
        assert len(lose_shifts) >= 3

        ws_spread = max(win_stays) - min(win_stays)
        ls_spread = max(lose_shifts) - min(lose_shifts)
        assert ws_spread < 0.5, f"Win-stay spread: {ws_spread:.2f}, values={win_stays}"
        assert ls_spread < 0.5, f"Lose-shift spread: {ls_spread:.2f}, values={lose_shifts}"

    def test_all_seeds_produce_metrics(self, multi_seed_metrics: list[dict]) -> None:
        """Every seed should produce valid metrics (no crashes)."""
        for i, m in enumerate(multi_seed_metrics):
            assert "psychometric" in m, f"Seed {SEEDS[i]}: missing psychometric"
            assert "history" in m, f"Seed {SEEDS[i]}: missing history"
            assert "quality" in m, f"Seed {SEEDS[i]}: missing quality"
