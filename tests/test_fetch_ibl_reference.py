from __future__ import annotations

import json

import numpy as np

from eval.schema_validator import validate_file
from scripts.fetch_ibl_reference import (
    calibrate_choice_sign,
    in_biased_block_contrast_set,
    session_to_records,
)


def _synthetic_session(n: int, right_choice_value: float, seed: int = 0) -> dict:
    """Build an IBL-like trials dict with a known choice-sign convention.

    ``right_choice_value`` is the raw ``choice`` value that corresponds to a
    rightward decision, letting tests inject either IBL sign convention.
    """
    rng = np.random.default_rng(seed)
    abs_c = rng.choice([0.0, 0.0625, 0.125, 0.25, 1.0, 0.5], size=n)  # 0.5 is off-protocol
    side = rng.choice([-1, 1], size=n)
    signed = abs_c * side
    cl = np.where(signed < 0, np.abs(signed), np.nan)
    cr = np.where(signed > 0, signed, np.nan)
    cr = np.where(np.isclose(signed, 0.0), 0.0, cr)
    correct = rng.random(n) < (0.5 + 0.45 * np.clip(abs_c, 0.0, 1.0))
    feedback = np.where(correct, 1, -1).astype(float)
    stim_is_right = signed > 0
    chose_right = np.where(correct, stim_is_right, ~stim_is_right)
    choice = np.where(chose_right, right_choice_value, -right_choice_value).astype(float)
    zero = np.isclose(signed, 0.0)
    choice[zero] = rng.choice([-1.0, 1.0], size=int(zero.sum()))
    stim_on = np.cumsum(rng.uniform(2.0, 4.0, size=n))
    first_move = stim_on + rng.uniform(0.15, 0.6, size=n)
    response = stim_on + rng.uniform(0.8, 1.5, size=n)
    prob_left = np.full(n, 0.5)
    prob_left[n // 3 : 2 * n // 3] = 0.2
    return {
        "contrastLeft": cl,
        "contrastRight": cr,
        "choice": choice,
        "feedbackType": feedback,
        "stimOn_times": stim_on,
        "firstMovement_times": first_move,
        "response_times": response,
        "probabilityLeft": prob_left,
    }


def test_contrast_filter_excludes_off_protocol_levels() -> None:
    assert in_biased_block_contrast_set(0.0)
    assert in_biased_block_contrast_set(-0.25)
    assert in_biased_block_contrast_set(1.0)
    assert not in_biased_block_contrast_set(0.5)  # not in biased-blocks protocol
    assert not in_biased_block_contrast_set(None)


def test_choice_sign_calibration_recovers_both_conventions() -> None:
    for truth in (-1.0, 1.0):
        trials = _synthetic_session(500, right_choice_value=truth)
        detected, agreement, n_used = calibrate_choice_sign(trials, 500)
        assert detected == truth
        assert agreement > 0.99
        assert n_used > 0


def test_unreliable_choice_column_reports_low_agreement() -> None:
    trials = _synthetic_session(500, right_choice_value=1.0)
    trials["choice"] = np.random.default_rng(3).choice([-1.0, 1.0], size=500)
    _, agreement, _ = calibrate_choice_sign(trials, 500)
    assert agreement < 0.7  # a random choice column carries no side information


def test_session_to_records_is_schema_valid_and_filtered(tmp_path) -> None:
    trials = _synthetic_session(400, right_choice_value=-1.0)  # inverted convention
    records, summary = session_to_records(trials, "synthetic-eid", rt_source="firstMovement")

    # Only biased-blocks contrasts survive.
    abs_contrasts = sorted({round(abs(r["stimulus"]["contrast"]), 4) for r in records})
    assert abs_contrasts == [0.0, 0.0625, 0.125, 0.25, 1.0]
    assert summary["dropped_off_protocol_contrast"] > 0
    # The inverted convention is detected, not silently trusted.
    assert summary["choice_sign_matches_legacy_assumption"] is False
    # trial_index is contiguous and prev is threaded within the session.
    assert [r["trial_index"] for r in records] == list(range(len(records)))
    assert records[0]["prev"] is None and records[1]["prev"] is not None

    log_path = tmp_path / "expanded.ndjson"
    log_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    result = validate_file(log_path, raise_on_error=False)
    assert result.errors == []
    assert result.total == len(records)
