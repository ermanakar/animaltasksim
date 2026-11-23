"""The outcome split must agree with the frozen models and recover a planted error-only effect."""
from __future__ import annotations

import pytest

from eval.bias_history import BiasConfig, evaluate_bias
from eval.competing_history import ComparisonConfig, compare, make_folds, prepare_comparison
from scripts.diagnose_bias_history import simulate
from scripts.split_evidence_by_outcome import evaluate_split


def test_split_matches_frozen_models_and_finds_planted_error_term() -> None:
    raw = simulate("history_only", 710)
    config = ComparisonConfig(folds=3)
    folds = make_folds(raw, config)
    gains = evaluate_split(raw, folds, BiasConfig())["gains"]

    # Adding both terms must reproduce the frozen evidence comparisons exactly.
    _, frozen_bias = evaluate_bias(raw, folds, BiasConfig())
    assert gains["bias_long"]["both"]["mean_gain"] == pytest.approx(
        frozen_bias["comparisons"]["bias_long_evidence_over_bias_long"]["mean_gain"], abs=1e-9)
    _, frozen = compare(prepare_comparison(raw, config), config, folds)
    assert gains["combined_history"]["both"]["mean_gain"] == pytest.approx(
        frozen["contrasts"]["combined_evidence_over_combined_history"]["mean_gain"], abs=1e-9)

    # The generator has a weak-evidence term after errors only.
    for control in ("combined_history", "bias_long"):
        assert gains[control]["error_unique"]["mean_gain"] > 0
        assert gains[control]["error_unique"]["mean_gain"] > gains[control]["reward_unique"]["mean_gain"]
