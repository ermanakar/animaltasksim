"""Independently reconcile an adopted IBL session against public ALF trial arrays."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eval.schema_validator import TrialRecord

__all__ = ["reconcile_session"]


def reconcile_session(
    records: list[dict[str, Any]], arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Verify source signals and produce a separate omission-corrected candidate session."""
    fields = ("choice", "feedbackType", "contrastLeft", "contrastRight",
              "probabilityLeft", "stimOn_times", "response_times")
    n = len(records)
    if any(key not in arrays or len(arrays[key]) != n for key in fields):
        raise ValueError("Source length/field mismatch; original row alignment is unverified.")
    if [row["trial_index"] for row in records] != list(range(n)):
        raise ValueError("Adopted source must have complete original trial indices.")
    if len({row["session_id"] for row in records}) != 1:
        raise ValueError("Reconcile exactly one session at a time.")
    choices = np.asarray(arrays["choice"])
    feedback = np.asarray(arrays["feedbackType"])
    if not np.isin(choices, [-1, 0, 1]).all() or not np.isin(feedback, [-1, 1]).all():
        raise ValueError("Unrecognized ALF choice or outcome coding.")
    # ALF choice denotes wheel turn: -1 selects a right-side stimulus, +1 left.
    # This is independent of the importer's feedback-based side inference.
    expected_actions = np.where(choices == -1, 0, np.where(choices == 1, 1, 2))
    cl, cr = np.asarray(arrays["contrastLeft"]), np.asarray(arrays["contrastRight"])
    if not np.logical_xor(np.isfinite(cl), np.isfinite(cr)).all():
        raise ValueError("Ambiguous or missing stimulus side.")
    contrast = np.where(np.isfinite(cr), cr, -cl)
    prior = 1 - np.asarray(arrays["probabilityLeft"])
    if not np.isfinite(prior).all():
        raise ValueError("Missing block priors.")
    correct = feedback == 1
    if np.any((choices == 0) & correct):
        raise ValueError("No-go trial is marked rewarded.")
    nonzero = (np.abs(contrast) > 1e-8) & (choices != 0)
    source_correct = (expected_actions == 0) == (contrast > 0)
    if np.any(source_correct[nonzero] != correct[nonzero]):
        raise ValueError("ALF choice sign disagrees with stimulus/feedback.")
    candidate, omission_indices = [], []
    previous = None
    for i, row in enumerate(records):
        if not np.isclose(row["stimulus"]["contrast"], contrast[i], atol=1e-8, rtol=0):
            raise ValueError(f"Stimulus mismatch at trial {i}.")
        if not np.isclose(row["block_prior"]["p_right"], prior[i], atol=1e-8, rtol=0):
            raise ValueError(f"Block prior mismatch at trial {i}.")
        if row["correct"] != bool(correct[i]) or row["reward"] != float(correct[i]):
            raise ValueError(f"Outcome mismatch at trial {i}.")
        elapsed = float((arrays["response_times"][i] - arrays["stimOn_times"][i]) * 1000)
        expected_rt = elapsed if np.isfinite(elapsed) and elapsed > 0 else None
        if (row["rt_ms"] is None) != (expected_rt is None) or (
            expected_rt is not None and not np.isclose(row["rt_ms"], expected_rt, atol=1e-6, rtol=0)
        ):
            raise ValueError(f"Response-time mismatch at trial {i}.")
        # Validate the adopted predecessor too; don't conceal unrelated old drift.
        old_previous = None if i == 0 else {key: records[i-1][key] for key in ("action", "reward", "correct")}
        if row["prev"] != old_previous:
            raise ValueError(f"Adopted history mismatch at trial {i}.")
        if row["action"] != int(expected_actions[i]):
            if choices[i] != 0:
                raise ValueError(f"Committed source choice mismatch at trial {i}.")
            omission_indices.append(i)
        new = copy.deepcopy(row)
        new["action"] = int(expected_actions[i])
        new["prev"] = previous
        if choices[i] == 0:
            new["rt_ms"] = None
        new["agent"]["version"] = "ibl_public_alf_reconciled_20260906"
        TrialRecord.model_validate(new)
        candidate.append(new)
        previous = {key: new[key] for key in ("action", "reward", "correct")}
    return candidate, {
        "n_trials": n, "raw_omissions": int(np.sum(choices == 0)),
        "misclassified_omissions": len(omission_indices), "affected_indices": omission_indices,
        "committed_choice_mismatches": 0, "source_signal_mismatches": 0,
        "adjacency_verified": True, "rt_convention": "response_times - stimOn_times; omissions have null RT",
    }
