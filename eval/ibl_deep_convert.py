"""Whole-session conversion of public IBL ALF trial arrays into schema trials, without quality filtering."""

from __future__ import annotations

from typing import Any

import numpy as np

from eval.schema_validator import TrialRecord

__all__ = ["ConversionError", "FAMILY_CONTRASTS", "REQUIRED", "convert_session", "compare_sessions"]

REQUIRED = ("choice", "feedbackType", "contrastLeft", "contrastRight",
            "probabilityLeft", "stimOn_times", "response_times")
_STANDARD = (0.0, 0.0625, 0.125, 0.25, 1.0)
FAMILY_CONTRASTS = {"biased": _STANDARD, "ephys": _STANDARD, "training": (*_STANDARD, 0.5)}
# Thresholds frozen for the September replication; recorded as a flag, never applied here.
_GATE_TRIALS, _GATE_EASY_TRIALS, _GATE_EASY_ACCURACY = 150, 20, 0.85


class ConversionError(ValueError):
    """A session that fails a structural rule; the message is the first failing rule."""


def convert_session(arrays: dict[str, np.ndarray], session_id: str,
                    family: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert every trial of one session or raise; trials are never dropped."""
    if family not in FAMILY_CONTRASTS:
        raise ValueError(f"Unknown session family {family}")
    missing = [key for key in REQUIRED if key not in arrays]
    if missing:
        raise ConversionError("missing_arrays")
    n = len(arrays["choice"])
    if any(len(arrays[key]) != n for key in REQUIRED):
        raise ConversionError("length_mismatch")
    if n == 0:
        raise ConversionError("empty")
    choice, feedback = np.asarray(arrays["choice"]), np.asarray(arrays["feedbackType"])
    if not np.isin(choice, [-1, 0, 1]).all() or not np.isin(feedback, [-1, 1]).all():
        raise ConversionError("unknown_choice_or_feedback_code")
    left, right = np.asarray(arrays["contrastLeft"], float), np.asarray(arrays["contrastRight"], float)
    prior = 1 - np.asarray(arrays["probabilityLeft"], float)
    if not np.logical_xor(np.isfinite(left), np.isfinite(right)).all() or not np.isfinite(prior).all():
        raise ConversionError("ambiguous_side_or_missing_prior")
    # Adding zero turns the -0.0 of a left-side blank into 0.0.
    contrast = np.where(np.isfinite(right), right, -left) + 0.0
    correct = feedback == 1
    if np.any((choice == 0) & correct):
        raise ConversionError("rewarded_omission")
    # ALF choice is the wheel turn: -1 selects the right stimulus, +1 the left.
    action = np.where(choice == -1, 0, np.where(choice == 1, 1, 2))
    informative = (contrast != 0) & (choice != 0)
    if np.any(((action == 0) == (contrast > 0))[informative] != correct[informative]):
        raise ConversionError("choice_sign_disagrees_with_feedback")
    allowed = np.asarray(FAMILY_CONTRASTS[family])
    if not np.isclose(np.abs(contrast)[:, None], allowed[None, :], atol=1e-8, rtol=0).any(axis=1).all():
        raise ConversionError("contrast_outside_protocol")
    with np.errstate(invalid="ignore"):
        elapsed = (np.asarray(arrays["response_times"], float) - np.asarray(arrays["stimOn_times"], float)) * 1000
    timed = (choice != 0) & np.isfinite(elapsed) & (elapsed > 0)
    rows, previous = [], None
    for i in range(n):
        row = {"task": "IBL2AFC", "session_id": session_id, "trial_index": i,
               "stimulus": {"contrast": float(contrast[i])}, "block_prior": {"p_right": float(prior[i])},
               "action": int(action[i]), "correct": bool(correct[i]), "reward": float(correct[i]),
               "rt_ms": float(elapsed[i]) if timed[i] else None, "phase_times": {}, "prev": previous, "seed": 0,
               "agent": {"name": "reference_mouse", "version": f"ibl_public_alf_deep_v1_{family}"}}
        TrialRecord.model_validate(row)
        rows.append(row)
        previous = {key: row[key] for key in ("action", "reward", "correct")}
    easy = np.abs(contrast) == 1
    committed = int((choice != 0).sum())
    accuracy = float(correct[easy].mean()) if easy.any() else None
    return rows, {
        "trials": n, "omissions": n - committed,
        "rt_coverage": float(timed.sum() / committed) if committed else None,
        "easy_trials": int(easy.sum()), "easy_accuracy": accuracy,
        "meets_frozen_gate": bool(n >= _GATE_TRIALS and easy.sum() >= _GATE_EASY_TRIALS
                                  and accuracy is not None and accuracy >= _GATE_EASY_ACCURACY)}


def compare_sessions(converted: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """List every disagreement with a reference session, ignoring only the agent version."""
    if len(converted) != len(reference):
        return [{"trial_index": None, "field": "length", "converted": len(converted), "reference": len(reference)}]
    differences = []
    for new, old in zip(converted, reference):
        for field in old:
            a, b = new.get(field), old[field]
            if field == "agent":
                a, b = a["name"], b["name"]
            same = a == b
            if field == "rt_ms" and a is not None and b is not None:
                same = bool(np.isclose(a, b, atol=1e-6, rtol=0))
            if not same or set(new) != set(old):
                differences.append({"trial_index": old["trial_index"], "field": field, "converted": a, "reference": b})
    return differences
