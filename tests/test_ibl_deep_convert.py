"""Structural conversion rules for the deep reference: whole sessions, verified coding, no filtering."""
from __future__ import annotations

import numpy as np
import pytest

from eval.ibl_deep_convert import ConversionError, compare_sessions, convert_session
from eval.ibl_source_audit import reconcile_session

NAN = np.nan


def _arrays() -> dict[str, np.ndarray]:
    """Right hit, left hit, omission, left-blank miss, right error with no stimulus-onset time."""
    return {"choice": np.array([-1, 1, 0, -1, 1]), "feedbackType": np.array([1, 1, -1, -1, -1]),
            "contrastLeft": np.array([NAN, 0.25, NAN, 0.0, NAN]), "contrastRight": np.array([1.0, NAN, 0.125, NAN, 1.0]),
            "probabilityLeft": np.array([0.5, 0.5, 0.2, 0.2, 0.2]),
            "stimOn_times": np.array([1.0, 5.0, 9.0, 13.0, NAN]), "response_times": np.array([1.3, 5.5, 69.0, 13.2, 17.4])}


def test_fields_follow_the_reconciled_reference_convention() -> None:
    rows, summary = convert_session(_arrays(), "s", "biased")
    assert [r["action"] for r in rows] == [0, 1, 2, 0, 1]
    assert [r["stimulus"]["contrast"] for r in rows] == [1.0, -0.25, 0.125, 0.0, 1.0]
    assert str(rows[3]["stimulus"]["contrast"]) == "0.0"
    assert rows[2]["rt_ms"] is None and rows[4]["rt_ms"] is None
    assert rows[1]["rt_ms"] == pytest.approx(500.0)
    assert rows[2]["block_prior"]["p_right"] == pytest.approx(0.8)
    assert rows[0]["prev"] is None and rows[3]["prev"] == {"action": 2, "reward": 0.0, "correct": False}
    assert summary == {"trials": 5, "omissions": 1, "rt_coverage": 0.75, "easy_trials": 2,
                       "easy_accuracy": 0.5, "meets_frozen_gate": False}


def test_conversion_agrees_with_independent_reconciliation() -> None:
    arrays = _arrays()
    arrays["stimOn_times"][4] = 17.0
    rows, _ = convert_session(arrays, "s", "biased")
    adopted = [dict(r, agent=dict(r["agent"])) for r in rows]
    adopted[2]["rt_ms"] = 60000.0  # the adopted reference kept a response time on omissions
    candidate, _ = reconcile_session(adopted, arrays)
    assert compare_sessions(rows, candidate) == []
    candidate[1]["action"] = 0
    assert [d["field"] for d in compare_sessions(rows, candidate)] == ["action"]
    assert compare_sessions(rows[:-1], candidate)[0]["field"] == "length"


@pytest.mark.parametrize(("change", "reason"), [
    (lambda a: a.pop("stimOn_times"), "missing_arrays"),
    (lambda a: a.update(choice=a["choice"][:-1]), "length_mismatch"),
    (lambda a: a.update(choice=np.array([-1, 1, 0, -1, 2])), "unknown_choice_or_feedback_code"),
    (lambda a: a.update(contrastLeft=np.array([0.5, 0.25, NAN, 0.0, NAN])), "ambiguous_side_or_missing_prior"),
    (lambda a: a.update(feedbackType=np.array([1, 1, 1, -1, -1])), "rewarded_omission"),
    (lambda a: a.update(choice=np.array([1, 1, 0, -1, 1])), "choice_sign_disagrees_with_feedback"),
    (lambda a: a.update(contrastLeft=np.array([NAN, 0.5, NAN, 0.0, NAN])), "contrast_outside_protocol"),
])
def test_structural_failures_reject_the_whole_session(change, reason: str) -> None:
    arrays = _arrays()
    change(arrays)
    with pytest.raises(ConversionError, match=reason):
        convert_session(arrays, "s", "biased")


def test_training_family_allows_half_contrast_and_records_priors_as_given() -> None:
    arrays = _arrays()
    arrays["contrastLeft"][1] = 0.5
    arrays["probabilityLeft"][0] = 0.3
    rows, _ = convert_session(arrays, "s", "training")
    assert rows[1]["stimulus"]["contrast"] == -0.5
    assert rows[0]["block_prior"]["p_right"] == pytest.approx(0.7)
    assert rows[0]["agent"]["version"].endswith("training")
