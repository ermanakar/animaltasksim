"""Causal feature construction and recovery checks for competing history models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from eval.competing_history import ComparisonConfig, compare, comparison_design, make_folds, prepare_comparison


def _trials(count: int = 24, session: str = "s0", subject: str = "m0") -> pd.DataFrame:
    """Make committed source-like trials with controlled feedback."""
    return pd.DataFrame({"task": "ibl_2afc", "session_id": session,
                         "subject": subject, "lab": "lab0", "trial_index": np.arange(count),
                         "action": "right", "correct": True, "stimulus_contrast": .25,
                         "block_prior": [{"p_right": .5} for _ in range(count)]})


def test_exact_lag_alignment_and_common_sample() -> None:
    raw = _trials(12)
    raw.loc[2, ["action", "correct"]] = ["left", False]
    data = prepare_comparison(raw, ComparisonConfig())
    assert data.trial_index.tolist() == list(range(5, 12))
    row = data[data.trial_index == 5].iloc[0]
    assert row.unrewarded_lag_3 == -1
    assert row.rewarded_lag_3 == 0
    assert row.rewarded_lag_2 == 1
    base, names = comparison_design(data, "combined_history", ComparisonConfig())
    extended, extended_names = comparison_design(data, "combined_evidence", ComparisonConfig())
    assert extended.shape[0] == base.shape[0]
    assert set(extended_names) - set(names) == {"rewarded_choice_weak", "unrewarded_choice_weak"}
    for i, name in enumerate(names):
        np.testing.assert_array_equal(base[:, i], extended[:, extended_names.index(name)])


def test_current_and_future_outcomes_cannot_change_current_predictors() -> None:
    config = ComparisonConfig()
    raw = _trials()
    altered = raw.copy(deep=True)
    altered.loc[12:, "action"] = "left"
    altered.loc[12:, "correct"] = False
    before, after = prepare_comparison(raw, config), prepare_comparison(altered, config)
    x, names = comparison_design(before[before.trial_index <= 12], "combined_evidence", config)
    changed, changed_names = comparison_design(after[after.trial_index <= 12], "combined_evidence", config)
    assert names == changed_names
    np.testing.assert_array_equal(x, changed)


@pytest.mark.parametrize("break_kind", ["omission", "gap"])
def test_omissions_and_gaps_reset_states_without_bridging(break_kind: str) -> None:
    raw = _trials()
    if break_kind == "omission":
        raw.loc[10, "action"] = "hold"
    else:
        raw = raw.drop(index=10)
    config = ComparisonConfig()
    data = prepare_comparison(raw, config)
    assert not data.trial_index.isin(range(10, 16)).any()
    row = data[data.trial_index == 16].iloc[0]
    for rate in config.rates:
        assert row[f"learned_side_{rate}"] == pytest.approx(1 - (1 - rate)**5)


def test_states_reset_between_sessions_and_hold_on_zero_contrast() -> None:
    raw = pd.concat([_trials(12), _trials(12, "s1")], ignore_index=True)
    raw.loc[(raw.session_id == "s1") & (raw.trial_index == 4), "stimulus_contrast"] = 0
    data = prepare_comparison(raw, ComparisonConfig())
    for rate in ComparisonConfig().rates:
        start = data[(data.session_id == "s1") & (data.trial_index == 5)].iloc[0]
        assert start[f"learned_side_{rate}"] == pytest.approx(1 - (1 - rate)**4)


def test_true_block_does_not_enter_learned_state() -> None:
    raw = _trials()
    altered = raw.copy(deep=True)
    altered["block_prior"] = [{"p_right": .8} for _ in range(len(raw))]
    first = prepare_comparison(raw, ComparisonConfig())
    second = prepare_comparison(altered, ComparisonConfig())
    for rate in ComparisonConfig().rates:
        np.testing.assert_array_equal(first[f"learned_side_{rate}"], second[f"learned_side_{rate}"])
    assert not first.prior_right.equals(second.prior_right)


def test_folds_keep_all_sessions_of_an_animal_together() -> None:
    raw = pd.concat([_trials(12, f"s{i}", f"m{i // 2}") for i in range(12)], ignore_index=True)
    config = ComparisonConfig(folds=3)
    data = prepare_comparison(raw, config)
    folds = make_folds(data, config)
    assert folds == make_folds(data, config)
    assert sorted(subject for fold in folds for subject in fold) == [f"m{i}" for i in range(6)]
    for heldout in folds:
        test = data.subject.isin(heldout)
        assert set(data.loc[test, "session_id"]).isdisjoint(data.loc[~test, "session_id"])
    with pytest.raises(ValueError, match="exactly once"):
        compare(data, config, folds + [folds[0]])


def _synthetic(evidence_weight: float) -> pd.DataFrame:
    """Generate causal binary decisions with a real lag-three effect and optional evidence interaction."""
    rng = np.random.default_rng(734)
    rows = []
    for animal in range(12):
        actions: list[int] = []
        successes: list[int] = []
        strengths: list[float] = []
        for trial in range(400):
            contrast = rng.choice([-.25, -.0625, .0625, .25])
            score = 1.5 * contrast
            if trial:
                score += .4 * actions[-1] * successes[-1]
                score += evidence_weight * actions[-1] * (1 - successes[-1]) * (strengths[-1] <= .125)
            if trial >= 3:
                score += 1.6 * actions[-3] * successes[-3]
            action = 1 if rng.random() < expit(score) else -1
            correct = int(action == np.sign(contrast))
            rows.append(dict(task="ibl_2afc", session_id=f"s{animal}", subject=f"m{animal}",
                             lab=f"lab{animal % 3}", trial_index=trial,
                             stimulus_contrast=contrast, action="right" if action == 1 else "left",
                             correct=bool(correct), block_prior={"p_right": .5}))
            actions.append(action)
            successes.append(correct)
            strengths.append(abs(contrast))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("evidence_weight", [0., 2.5])
def test_recovers_long_history_and_additional_evidence(evidence_weight: float) -> None:
    config = ComparisonConfig(folds=3)
    data = prepare_comparison(_synthetic(evidence_weight), config)
    predictions, summary = compare(data, config, make_folds(data, config))
    nll = summary["equal_animal_nll"]
    assert nll["outcome_history"] - nll["long_history"] > .02
    gain = summary["contrasts"]["combined_evidence_over_combined_history"]["mean_gain"]
    if evidence_weight:
        assert gain > .015
    else:
        assert abs(gain) < .003
    # Every animal is predicted by exactly one fold for each model.
    assert predictions.groupby(["subject", "model"]).fold.nunique().eq(1).all()
