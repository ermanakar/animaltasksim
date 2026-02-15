from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.metrics import (
    compute_chronometric,
    compute_history_metrics,
    compute_psychometric,
)


def _make_psychometric_df() -> pd.DataFrame:
    rows = []
    task = "ibl_2afc"
    trial_index = 0
    for contrast, right_ratio in [(-1.0, 0.1), (-0.5, 0.3), (0.5, 0.7), (1.0, 0.9)]:
        total = 40
        right = int(total * right_ratio)
        left = total - right
        rows.extend(
            {
                "task": task,
                "session_id": "s1",
                "trial_index": trial_index + i,
                "stimulus_contrast": contrast,
                "action": "right",
                "reward": 1.0 if contrast > 0 else 0.0,
                "rt_ms": 500.0,
                "prev_action": "right" if i > 0 else None,
                "prev_reward": 1.0 if i > 0 and contrast > 0 else 0.0,
                "prev_correct": 1.0 if i > 0 and contrast > 0 else 0.0,
            }
            for i in range(right)
        )
        trial_index += right
        rows.extend(
            {
                "task": task,
                "session_id": "s1",
                "trial_index": trial_index + i,
                "stimulus_contrast": contrast,
                "action": "left",
                "reward": 1.0 if contrast < 0 else 0.0,
                "rt_ms": 500.0,
                "prev_action": "left" if i > 0 else None,
                "prev_reward": 1.0 if i > 0 and contrast < 0 else 0.0,
                "prev_correct": 1.0 if i > 0 and contrast < 0 else 0.0,
            }
            for i in range(left)
        )
        trial_index += left
    return pd.DataFrame(rows)


def _make_chronometric_df() -> pd.DataFrame:
    rows = []
    task = "rdm"
    session = "s2"
    idx = 0
    for coherence in [0.1, 0.3, 0.6]:
        for _ in range(20):
            rt = 300.0 + 200.0 * coherence
            rows.append(
                {
                    "task": task,
                    "session_id": session,
                    "trial_index": idx,
                    "stimulus_coherence": coherence,
                    "action": "right",
                    "reward": 1.0,
                    "rt_ms": rt,
                "prev_action": "right" if idx > 0 else None,
                "prev_reward": 1.0 if idx > 0 else None,
                "prev_correct": 1.0 if idx > 0 else None,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def _make_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "ibl_2afc",
                "session_id": "s3",
                "trial_index": 0,
                "stimulus_contrast": -0.5,
                "action": "left",
                "reward": 1.0,
                "rt_ms": 520.0,
                "prev_action": None,
                "prev_reward": None,
                "prev_correct": None,
            },
            {
                "task": "ibl_2afc",
                "session_id": "s3",
                "trial_index": 1,
                "stimulus_contrast": -0.5,
                "action": "left",
                "reward": 1.0,
                "rt_ms": 500.0,
                "prev_action": "left",
                "prev_reward": 1.0,
                "prev_correct": 1.0,
            },
            {
                "task": "ibl_2afc",
                "session_id": "s3",
                "trial_index": 2,
                "stimulus_contrast": 0.5,
                "action": "right",
                "reward": 1.0,
                "rt_ms": 480.0,
                "prev_action": "left",
                "prev_reward": 0.0,
                "prev_correct": 0.0,
            },
            {
                "task": "ibl_2afc",
                "session_id": "s3",
                "trial_index": 3,
                "stimulus_contrast": 0.5,
                "action": "right",
                "reward": 1.0,
                "rt_ms": 460.0,
                "prev_action": "right",
                "prev_reward": 1.0,
                "prev_correct": 1.0,
            },
        ]
    )


def test_psychometric_fit_returns_positive_slope():
    df = _make_psychometric_df()
    metrics = compute_psychometric(df, stimulus_key="contrast")
    assert metrics.slope > 0
    assert abs(metrics.bias) < 0.3


def test_chronometric_slope_matches_trend():
    df = _make_chronometric_df()
    metrics = compute_chronometric(df, stimulus_key="coherence")
    assert metrics.slope_ms_per_unit == pytest.approx(20.0, rel=1e-2)
    assert metrics.intercept_ms == pytest.approx(300.0, rel=1e-2)
    assert metrics.slope_unit == "ms_per_10pct_coherence"


def test_history_metrics_win_stay_and_betas():
    df = _make_history_df()
    metrics = compute_history_metrics(df)
    assert metrics.win_stay > 0.5
    assert metrics.sticky_choice > 0.5
    assert np.isfinite(metrics.prev_choice_beta)
    assert np.isfinite(metrics.prev_correct_beta)


def _make_ceiling_df() -> pd.DataFrame:
    """RT data where low-coherence trials are pinned at a ceiling value."""
    rows = []
    idx = 0
    ceiling_rt = 1200.0
    for coherence, rt in [(0.0, ceiling_rt), (0.032, ceiling_rt), (0.064, ceiling_rt),
                          (0.128, 880.0), (0.256, 540.0), (0.512, 370.0)]:
        for _ in range(20):
            rows.append({
                "task": "rdm",
                "session_id": "s4",
                "trial_index": idx,
                "stimulus_coherence": coherence,
                "action": "right",
                "reward": 1.0,
                "rt_ms": rt,
                "prev_action": "right" if idx > 0 else None,
                "prev_reward": 1.0 if idx > 0 else None,
                "prev_correct": 1.0 if idx > 0 else None,
            })
            idx += 1
    return pd.DataFrame(rows)


def test_ceiling_detection_flags_saturated_rts():
    """When ≥50% of difficulty levels are pinned at max RT, ceiling_fraction is high."""
    df = _make_ceiling_df()
    metrics = compute_chronometric(df, stimulus_key="coherence")
    # 3 of 6 levels at ceiling = 0.5
    assert metrics.ceiling_fraction >= 0.5, (
        f"Expected ceiling_fraction >= 0.5, got {metrics.ceiling_fraction}"
    )
    assert metrics.rt_range_ms > 0, "RT range should be non-zero for mixed RTs"


def test_no_ceiling_for_graded_rts():
    """When RTs are smoothly graded, ceiling_fraction should be low."""
    df = _make_chronometric_df()
    metrics = compute_chronometric(df, stimulus_key="coherence")
    # All RTs are different, none at ceiling
    assert metrics.ceiling_fraction < 0.5, (
        f"Expected low ceiling_fraction, got {metrics.ceiling_fraction}"
    )
    # No corrected slope needed when no ceiling
    assert metrics.corrected_slope is None


def test_ceiling_corrected_slope():
    """Corrected slope excludes ceiling levels and refits."""
    df = _make_ceiling_df()
    metrics = compute_chronometric(df, stimulus_key="coherence")
    # Corrected slope should exist when ceiling is detected
    assert metrics.corrected_slope is not None
    # Corrected slope should be negative (easy = fast, hard = slow)
    assert metrics.corrected_slope < 0
    # Corrected slope should differ from raw slope since ceiling levels are excluded
    assert metrics.corrected_slope != metrics.slope_ms_per_unit
