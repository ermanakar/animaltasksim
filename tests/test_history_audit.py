"""Recovery and exclusion checks for the exploratory history comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from eval.history_audit import AuditConfig, audit_history, design_matrix, fit_logistic, prepare_trials


def _trials() -> pd.DataFrame:
    rows = []
    for session in ("a", "b", "c", "d"):
        for index in range(20):
            rows.append({"task": "ibl_2afc", "session_id": session, "trial_index": index,
                         "stimulus_contrast": 0.125 if index % 2 else -0.25,
                         "block_prior": {"p_right": 0.5}, "action": "right" if index % 3 else "left",
                         "correct": index % 3 != 0})
    return pd.DataFrame(rows)


def test_audit_never_bridges_missing_trials_omissions_or_sessions() -> None:
    raw = _trials()
    raw.loc[(raw.session_id == "a") & (raw.trial_index == 5), "action"] = "no-op"
    raw = raw[~((raw.session_id == "b") & (raw.trial_index == 5))]
    data = prepare_trials(raw, AuditConfig())
    assert not (data.trial_index == 0).any()
    assert not ((data.session_id == "a") & data.trial_index.isin([5, 6])).any()
    assert not ((data.session_id == "b") & (data.trial_index == 6)).any()
    assert len(data) == 72


def test_fold_membership_is_disjoint_complete_and_deterministic() -> None:
    config = AuditConfig(folds=2)
    result = audit_history(_trials(), config)
    folds = result["folds"]
    assert set(folds[0]).isdisjoint(folds[1])
    assert sorted(folds[0] + folds[1]) == ["a", "b", "c", "d"]
    assert result == audit_history(_trials().sample(frac=1, random_state=5), config)
    assert all(np.isfinite(list(result["pooled_nll"].values())))


@pytest.mark.parametrize("interaction", [0.0, 1.4])
def test_history_interaction_recovery_and_predictive_gain(interaction: float) -> None:
    """An independently generated weak-failure interaction must be recoverable."""
    rng = np.random.default_rng(123)
    n = 16000
    data = pd.DataFrame({
        "stimulus_contrast": rng.choice([-0.125, 0.125], n),
        "prior_right": rng.choice([0.2, 0.5, 0.8], n),
        "previous_correct": rng.integers(0, 2, n).astype(float),
        "previous_right_signed": rng.choice([-1.0, 1.0], n),
        "previous_strength": rng.choice([0.125, 1.0], n),
    })
    data["previous_weak"] = (data.previous_strength <= 0.125).astype(float)
    target = expit(4 * data.stimulus_contrast + 0.3 * data.previous_right_signed
                   + interaction * data.previous_right_signed * (1 - data.previous_correct) * data.previous_weak)
    y = (rng.random(n) < target).astype(float)
    losses = []
    for model in ("outcome_history", "evidence_history"):
        x, names = design_matrix(data, model)
        beta = fit_logistic(x[:12000], y[:12000], 1.0)
        z = x[12000:] @ beta
        losses.append(np.mean(np.logaddexp(0, z) - y[12000:] * z))
        if model == "evidence_history":
            assert beta[names.index("unrewarded_choice_weak")] == pytest.approx(interaction, abs=0.25)
    if interaction:
        assert losses[0] - losses[1] > 0.005
    else:
        assert losses[0] - losses[1] < 0.001


def test_audit_rejects_duplicate_trials_and_insufficient_sessions() -> None:
    raw = _trials()
    with pytest.raises(ValueError, match="unique"):
        prepare_trials(pd.concat([raw, raw.iloc[:1]]), AuditConfig())
    with pytest.raises(ValueError, match="Fewer"):
        audit_history(raw, AuditConfig(folds=5))


def test_subject_folds_never_split_sessions_from_the_same_mouse() -> None:
    subjects = {"a": "mouse1", "b": "mouse1", "c": "mouse2", "d": "mouse3"}
    result = audit_history(_trials(), AuditConfig(folds=2), session_subjects=subjects)
    fold_subjects = [{subjects[sid] for sid in fold} for fold in result["folds"]]
    assert fold_subjects[0].isdisjoint(fold_subjects[1])
    assert result["eligible_subjects"] == 3
    assert len(result["per_subject"]) == 3
    with pytest.raises(ValueError, match="Missing subject"):
        audit_history(_trials(), AuditConfig(folds=2), session_subjects={"a": "mouse1"})
