"""Independent checks of exploratory subject splitting and prediction scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.validate_architecture_prediction import baseline_predict, fit_baseline, paired_summary, score_rows, split_subjects


def test_split_is_subject_disjoint_and_order_invariant() -> None:
    subjects = [f"mouse-{i}" for i in range(25)]
    train, test = split_subjects(subjects, 12, 8, 42)
    assert not set(train) & set(test)
    assert (train, test) == split_subjects(subjects[::-1] + subjects, 12, 8, 42)
    with pytest.raises(ValueError):
        split_subjects(subjects, 24, 8, 42)


def test_simple_baseline_learns_predictive_stimulus() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(1000, 7))
    y = (rng.random(1000) < 1 / (1 + np.exp(-3 * features[:, 0]))).astype(float)
    rt = 900 + 100 * features[:, 1]
    fit = fit_baseline(features, y, np.ones(1000), rt, np.ones(1000))
    p, pred_rt = baseline_predict(fit, features)
    assert -(y * np.log(p) + (1-y)*np.log1p(-p)).mean() < .5
    assert np.mean((pred_rt - rt)**2) < 1


def test_scores_do_not_turn_omissions_into_left_choices() -> None:
    frame = pd.DataFrame(dict(subject=["a"]*3, session_id=["s"]*3, trial_index=[0,1,2],
                              action=["right", "left", "no_op"], rt_ms=[100, 4000, None]))
    scores = score_rows(frame, np.array([.8,.3,.99]), np.array([100,3000,100]), "m", 1)
    assert scores.choice_nll.iloc[0] == pytest.approx(-np.log(.8))
    assert np.isnan(scores.choice_nll.iloc[2])
    assert scores.rt_squared_error_seconds.iloc[1] == 1
    assert scores.rt_outside_window.sum() == 1
    assert scores.omission.sum() == 1


def test_summary_weights_animals_equally() -> None:
    rows = []
    for subject, count, gain in [("a", 100, 1), ("b", 1, -.5)]:
        for model in ["simple_history", "no_control", "full_control"]:
            rows += [dict(subject=subject, model=model, choice_nll=2 - (gain if model=="full_control" else 0),
                          rt_squared_error_seconds=2 - (gain if model=="full_control" else 0))] * count
    summary = paired_summary(pd.DataFrame(rows))
    assert summary['full_control_vs_simple_history']['choice_nll']['mean_improvement'] == .25
    assert not summary['full_control_vs_simple_history']['exploratory_gate_pass']
