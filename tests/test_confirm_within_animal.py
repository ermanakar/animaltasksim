from __future__ import annotations

import numpy as np
import pytest

from scripts.confirm_within_animal import CLAIMS, bounds, read_claims, select_cohort


def _session(index: int, subject: str, protocol: str = "_iblrig_tasks_biasedChoiceWorld6.4.2") -> dict:
    return {"id": f"{subject}-{index}", "subject": subject, "lab": "lab", "start_time": f"2020-01-{index + 1:02d}", "task_protocol": protocol}


def test_cohort_excludes_known_animals_variants_and_shallow_animals() -> None:
    frame = [_session(i, "new") for i in range(6)] + [_session(i, "seen") for i in range(8)]
    frame += [_session(i, "shallow") for i in range(5)] + [_session(i, "opto", "nate_optoBiasedChoiceWorld8.26.0") for i in range(9)]
    frame += [_session(i, "mixed") for i in range(5)] + [_session(9, "mixed", "_iblrig_tasks_trainingChoiceWorld6.4.2")]
    cohort = select_cohort(frame, {"seen"})
    assert {s["subject"] for s in cohort} == {"new"} and len(cohort) == 6
    assert all(s["family"] == "biased" for s in cohort)
    with pytest.raises(ValueError):
        select_cohort(frame + [_session(0, "new") | {"lab": "other"}], set())


def test_bounds_widen_with_the_level() -> None:
    values = np.random.default_rng(0).normal(1.0, 1.0, 40)
    narrow = bounds(values, np.random.default_rng(1), 4000, 0.95)
    wide = bounds(values, np.random.default_rng(1), 4000, 0.99)
    assert wide["low"][0] < narrow["low"][0] < narrow["mean"][0] < narrow["high"][0] < wide["high"][0]


def _inputs(score: float, state: float, rng: np.random.Generator, animals: int = 40) -> tuple[dict, dict]:
    gains = {name: rng.normal(score, 0.002, animals) for _, names, _ in CLAIMS.values() for name in names if name.startswith(("state", "history"))}
    values = {"p1_congruence": rng.normal(-state, 0.05, animals), "p2_magnitude": rng.normal(state, 0.05, animals),
              "p3_omission": rng.normal(state, 0.05, animals)}
    return gains, values


def test_claims_follow_the_protocol_rules() -> None:
    rng = np.random.default_rng(2)
    large = read_claims(*_inputs(0.01, 0.1, rng), rng, 2000)
    assert all(c["reading"] == "confirmed" for c in large.values())
    assert large["p1_congruence"]["quantities"]["p1_congruence"]["mean"] < 0  # reported in its original sign
    null = read_claims(*_inputs(0.0, 0.0, rng), rng, 2000)
    assert all(c["reading"] == "not confirmed" for c in null.values())
    reverse = read_claims(*_inputs(-0.01, -0.1, rng), rng, 2000)
    assert all(c["reading"] == "contradicted" for c in reverse.values())


def test_state_needs_one_contrast_and_history_needs_both() -> None:
    rng = np.random.default_rng(3)
    gains, values = _inputs(0.01, 0.1, rng)
    gains["state_switching"] = gains["state_switching"] * 0
    gains["history_switching"] = gains["history_switching"] * 0
    reading = read_claims(gains, values, rng, 2000)
    assert reading["c1_state_beyond_history"]["reading"] == "confirmed"
    assert reading["c2_history_beyond_state"]["reading"] == "not confirmed"


def test_state_claims_are_skipped_without_values_and_empty_quantities_do_not_confirm() -> None:
    rng = np.random.default_rng(4)
    gains, values = _inputs(0.01, 0.1, rng)
    assert set(read_claims(gains, None, rng, 500)) == {"c1_state_beyond_history", "c2_history_beyond_state"}
    assert read_claims(gains, values | {"p3_omission": np.zeros(0)}, rng, 500)["p3_omission"]["reading"] == "not confirmed"
