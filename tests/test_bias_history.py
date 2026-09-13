"""Independent causal and fold-isolation checks for prefix/slow-bias controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.bias_history import BiasConfig, bias_features, evaluate_bias, task_design


def _raw(subjects: int = 1, trials: int = 30) -> pd.DataFrame:
    """Create source-like trials with varied choices and task cells."""
    rng = np.random.default_rng(144)
    rows = []
    for subject in range(subjects):
        for trial in range(trials):
            contrast = float(rng.choice([-.25, -.0625, .0625, .25]))
            right = bool(rng.random() < .5)
            rows.append(dict(task="ibl_2afc", subject=f"m{subject}", session_id=f"s{subject}",
                             lab="lab", trial_index=trial, stimulus_contrast=contrast,
                             block_prior={"p_right": .5}, action="right" if right else "left",
                             correct=right == (contrast > 0)))
    return pd.DataFrame(rows)


def _features(raw: pd.DataFrame, config: BiasConfig) -> pd.DataFrame:
    return bias_features(raw, np.zeros(task_design(raw)[0].shape[1]), config)


def test_prefix_is_not_scored_or_exposed_early() -> None:
    raw = _raw()
    raw.loc[:9, "action"] = "right"
    config = BiasConfig(prefix=10, min_prefix_choices=8)
    data = _features(raw, config)
    assert not data.loc[:9, "after_prefix"].any()
    assert data.loc[:9, "prefix_bias"].eq(0).all()
    assert data.loc[10:, "after_prefix"].all()
    assert data.loc[10:, "prefix_bias"].gt(0).all()
    assert data.loc[10:, "prefix_bias"].nunique() == 1
    raw.loc[10:, "action"] = "left"
    changed = _features(raw, config)
    np.testing.assert_array_equal(data.prefix_bias, changed.prefix_bias)


def test_current_and_future_choices_do_not_enter_current_state() -> None:
    raw = _raw()
    config = BiasConfig(prefix=10, min_prefix_choices=8)
    first = _features(raw, config)
    changed_raw = raw.copy(deep=True)
    changed_raw.loc[18:, "action"] = np.where(changed_raw.loc[18:, "action"] == "right", "left", "right")
    changed_raw.loc[18:, "correct"] = ~changed_raw.loc[18:, "correct"]
    second = _features(changed_raw, config)
    for name in ["prefix_bias", *[f"residual_trace_{r}" for r in config.rates]]:
        np.testing.assert_array_equal(first.loc[:18, name], second.loc[:18, name])


@pytest.mark.parametrize("kind", ["omission", "gap"])
def test_residual_trace_resets_before_prediction_after_break(kind: str) -> None:
    raw = _raw()
    raw.action = "right"
    if kind == "omission":
        raw.loc[15, "action"] = "hold"
    else:
        raw = raw.drop(index=15)
    config = BiasConfig(prefix=10, min_prefix_choices=8)
    data = _features(raw, config).set_index("trial_index")
    for rate in config.rates:
        assert data.at[16, f"residual_trace_{rate}"] == 0
        assert data.at[17, f"residual_trace_{rate}"] == pytest.approx(rate * .5)


def test_prefix_choice_threshold_and_session_state_reset() -> None:
    raw = _raw(subjects=2)
    raw.loc[(raw.subject == "m0") & (raw.trial_index < 3), "action"] = "hold"
    config = BiasConfig(prefix=10, min_prefix_choices=8)
    data = _features(raw, config)
    assert not data.loc[data.subject == "m0", "after_prefix"].any()
    assert data.loc[(data.subject == "m1") & (data.trial_index >= 10), "after_prefix"].all()
    for rate in config.rates:
        assert data.loc[(data.subject == "m1") & (data.trial_index == 0), f"residual_trace_{rate}"].item() == 0


def test_test_responses_cannot_change_global_or_nested_fit_coefficients() -> None:
    raw = _raw(subjects=4, trials=40)
    config = BiasConfig(prefix=10, min_prefix_choices=8)
    folds = [["m0", "m1"], ["m2", "m3"]]
    predictions, result = evaluate_bias(raw, folds, config)
    altered = raw.copy(deep=True)
    heldout = altered.subject.isin(folds[0])
    altered.loc[heldout, "action"] = np.where(altered.loc[heldout, "action"] == "right", "left", "right")
    altered.loc[heldout, "correct"] = ~altered.loc[heldout, "correct"]
    _, altered_result = evaluate_bias(altered, folds, config)
    original_fit = result["fits"][0]
    changed_fit = altered_result["fits"][0]
    assert original_fit["task_coefficients"] == changed_fit["task_coefficients"]
    assert original_fit["models"] == changed_fit["models"]
    assert predictions.trial_index.ge(10).all()
    assert predictions.groupby(["subject", "model"]).fold.nunique().eq(1).all()
    for fold, members in enumerate(folds):
        assert set(predictions.loc[predictions.fold == fold, "subject"]) == set(members)
    keys = ["subject", "session_id", "trial_index"]
    reference = None
    for _, group in predictions.groupby("model"):
        sample = set(map(tuple, group[keys].to_numpy()))
        if reference is None:
            reference = sample
        assert sample == reference


def test_duplicate_folds_rejected() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_bias(_raw(subjects=2), [["m0"], ["m0", "m1"]], BiasConfig(prefix=10, min_prefix_choices=8))
