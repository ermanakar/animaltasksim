from __future__ import annotations

import json

import numpy as np

from eval.metrics import compute_psychometric, load_trials
from eval.schema_validator import validate_file
from scripts.fetch_ibl_reference import (
    _is_nan,
    calibrate_choice_sign,
    in_biased_block_contrast_set,
    session_to_records,
)


def test_is_nan_handles_numpy_and_non_numeric() -> None:
    import numpy as np

    # numpy float32 is NOT a Python float subclass — the naive isinstance check
    # missed it and let NaNs slip through signed_contrast/RT logic.
    assert _is_nan(np.float32("nan"))
    assert _is_nan(np.float64("nan"))
    assert _is_nan(float("nan"))
    assert not _is_nan(np.float32(0.0625))
    assert not _is_nan(0.25)
    assert not _is_nan(None)
    assert not _is_nan("left")


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


def test_metrics_pipeline_recovers_a_sane_psychometric(tmp_path) -> None:
    # End-to-end guard for the action-integer convention: reference logs use
    # 0=right/1=left (eval.metrics.load_trials), the inverse of the env. Emitting
    # the wrong convention inverts the curve and collapses the fit (slope->0,
    # lapse->0.5). A trained synthetic mouse must fit a steep, low-lapse sigmoid.
    trials = _synthetic_session(1200, right_choice_value=1.0)
    records, _ = session_to_records(trials, "trained", rt_source="firstMovement")
    log_path = tmp_path / "expanded.ndjson"
    log_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    metrics = compute_psychometric(load_trials(log_path), stimulus_key="contrast")
    assert metrics.slope > 3.0  # steep, not the ~0 of an inverted fit
    assert metrics.lapse_low < 0.2 and metrics.lapse_high < 0.2  # not the ~0.5 collapse


def test_qc_reports_trained_and_untrained_full_contrast_accuracy() -> None:
    import numpy as np

    # A trained mouse (accuracy scales with contrast) clears the QC gate.
    trained = _synthetic_session(600, right_choice_value=1.0)
    _, summary = session_to_records(trained, "trained", rt_source="firstMovement")
    assert summary["n_full_contrast"] >= 20
    assert summary["easy_full_contrast_accuracy"] >= 0.85

    # A near-chance session (feedback independent of contrast) fails QC.
    untrained = _synthetic_session(600, right_choice_value=1.0)
    n = len(untrained["feedbackType"])
    untrained["feedbackType"] = np.where(
        np.random.default_rng(7).random(n) < 0.5, 1.0, -1.0
    )
    _, summary_bad = session_to_records(untrained, "untrained", rt_source="firstMovement")
    assert summary_bad["easy_full_contrast_accuracy"] < 0.85
