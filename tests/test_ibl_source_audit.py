"""Source-reconciliation regressions with independently specified ALF trials."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from eval.ibl_source_audit import reconcile_session


def _source() -> tuple[list[dict], dict[str, np.ndarray]]:
    arrays = {"choice": np.array([-1, 0, 1]), "feedbackType": np.array([1, -1, -1]),
              "contrastLeft": np.array([np.nan, np.nan, np.nan]),
              "contrastRight": np.array([1.0, 1.0, 0.125]),
              "probabilityLeft": np.array([0.5, 0.5, 0.5]),
              "stimOn_times": np.array([0.0, 1.0, 70.0]),
              "response_times": np.array([0.4, 61.0, 70.4])}
    rows, previous = [], None
    for i, action in enumerate([0, 1, 1]):  # Historic importer mistakes no-go for left.
        correct = i == 0
        row = {"task": "IBL2AFC", "session_id": "s", "trial_index": i,
               "stimulus": {"contrast": float(arrays["contrastRight"][i])}, "block_prior": {"p_right": 0.5},
               "action": action, "correct": correct, "reward": float(correct),
               "rt_ms": float((arrays["response_times"][i]-arrays["stimOn_times"][i])*1000),
               "phase_times": {}, "prev": previous, "seed": 0,
               "agent": {"name": "reference_mouse", "version": "legacy"}}
        rows.append(row)
        previous = {key: row[key] for key in ("action", "reward", "correct")}
    return rows, arrays


def test_source_reconciliation_corrects_nogo_and_next_history_without_mutating_original() -> None:
    rows, arrays = _source()
    before = copy.deepcopy(rows)
    candidate, report = reconcile_session(rows, arrays)
    assert rows == before
    assert report["misclassified_omissions"] == 1
    assert candidate[1]["action"] == 2 and candidate[1]["rt_ms"] is None
    assert candidate[2]["prev"]["action"] == 2
    assert candidate[2]["action"] == 1


@pytest.mark.parametrize("field", ["choice", "contrastRight", "probabilityLeft", "response_times"])
def test_source_reconciliation_rejects_unexplained_drift(field: str) -> None:
    rows, arrays = _source()
    arrays[field][0] += 0.1
    with pytest.raises(ValueError):
        reconcile_session(rows, arrays)
