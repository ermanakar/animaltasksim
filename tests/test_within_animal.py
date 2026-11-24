"""Gradients, causality and recovery checks for the within-animal models and their simulator."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from eval import within_animal as wa
from eval import within_animal_recovery as wr


def _schedule(seed: int, sessions: int = 4, low: int = 60, high: int = 90, omit: float = 0.03) -> wa.Sessions:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(sessions):
        n = int(rng.integers(low, high))
        prior = np.r_[np.full(20, 0.5), np.repeat(np.tile([0.2, 0.8], 50), 30)][:n]
        rows.append({"contrast": rng.choice(wa.CONTRASTS, n), "prior": prior,
                     "choice": np.where(rng.random(n) < omit, 0.0, 1.0), "correct": np.zeros(n)})
    return wa.pad(rows)


def test_analytic_gradients_match_finite_differences() -> None:
    data = wa.design(wr.simulate(_schedule(0), "drift_history", np.random.default_rng(1))[0])
    x = np.concatenate([data.x, data.h], axis=2)
    rng = np.random.default_rng(2)
    theta, grid = np.r_[rng.normal(0, 0.5, x.shape[2]), 0.97, 0.6], wa._grid(15)
    numeric = approx_fprime(theta, lambda t: wa._drift_loss(t, x, data, grid)[0], 1e-6)
    assert np.allclose(numeric, wa._drift_loss(theta, x, data, grid)[1], atol=2e-3, rtol=1e-4)
    theta = np.r_[rng.normal(0, 0.5, 3 * x.shape[2]), rng.normal(0, 1, 9)]
    numeric = approx_fprime(theta, lambda t: wa._switching_loss(t, x, data)[0], 1e-6)
    assert np.allclose(numeric, wa._switching_loss(theta, x, data)[1], atol=1e-4)


def test_history_terms_are_causal_and_cleared_by_omissions() -> None:
    choice = np.array([[1.0, -1.0, 0.0, 1.0, 1.0, -1.0]])
    correct = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])
    contrast = np.array([[0.125, 1.0, 1.0, 0.0625, 1.0, 0.25]])
    h = wa.history_features(choice, correct, contrast)[0]
    assert not h[0].any() and not h[3].any()  # first trial, and the trial after an omission
    assert h[1, 0] == 1 and h[1, 10] == 1  # rewarded right choice on a weak trial
    assert h[2, 1] == -1 and h[2, 11] == 0 and h[2, 2] == 1  # unrewarded left on a strong trial; lag 2 kept
    assert h[5, 1] == 1 and h[5, 2] == 1 and not h[5, 4:10].any()  # history does not bridge the omission
    changed = choice.copy()
    changed[0, 4:] *= -1
    assert np.array_equal(wa.history_features(changed, correct, contrast)[0, :5], h[:5])


@pytest.mark.parametrize("model", ["drift_history", "switching_history"])
def test_predictions_ignore_later_trials_and_omissions_carry_no_choice(model: str) -> None:
    sessions, _ = wr.simulate(_schedule(3), "drift_history", np.random.default_rng(4))
    fitted = wa.fit(wa.design(sessions), model, maxiter=30, states=15)
    before = wa.predict(wa.design(sessions), fitted)
    altered = wa.Sessions(sessions.contrast, sessions.prior, sessions.choice.copy(), sessions.correct.copy(), sessions.length)
    altered.choice[:, 40:] *= -1
    after = wa.predict(wa.design(altered), fitted)
    assert np.allclose(before[:40], after[:40], equal_nan=True)
    assert np.isnan(before[~wa.design(sessions).committed]).all()


def test_simulator_and_design_agree_on_the_history_terms() -> None:
    sessions, truth = wr.simulate(_schedule(5, sessions=12, low=700, high=900), "history", np.random.default_rng(11))
    data = wa.design(sessions)
    levels = np.asarray(wa.CONTRASTS)
    task = np.r_[truth["bias"] + truth["sensitivity"] * 4 * np.sign(levels) * np.sqrt(np.abs(levels)), 0.4]
    weights = wr.history_weights(1.0) * truth["history_scale"]

    def loss(history: np.ndarray) -> float:
        z = data.x @ task + history @ weights
        return float(np.sum((np.logaddexp(0, z) - data.y * z) * data.committed) / data.committed.sum())

    exact, fitted = loss(data.h), float(np.nanmean(wa.predict(data, wa.fit(data, "history"))))
    assert abs(exact - fitted) < 0.003  # the generating parameters explain the choices as well as the fit
    shifted = np.roll(data.h, 1, axis=0)
    assert loss(shifted) > exact + 0.003 and loss(np.zeros_like(data.h)) > exact + 0.003


def test_fitted_drift_beats_fixed_only_when_drift_is_present() -> None:
    schedule = _schedule(7, sessions=8, low=500, high=600, omit=0.01)
    models = ("fixed", "drift")
    with_drift = wr.score_animal(wr.simulate(schedule, "drift", np.random.default_rng(8))[0], states=21, models=models)
    without = wr.score_animal(wr.simulate(schedule, "fixed", np.random.default_rng(8))[0], states=21, models=models)
    assert with_drift["nll"]["fixed"] - with_drift["nll"]["drift"] > 0.003
    assert abs(without["nll"]["fixed"] - without["nll"]["drift"]) < 0.001
    assert with_drift["drift"]["drift"]["rho"] > 0.97


def test_detection_rules_and_cohort_probabilities() -> None:
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 0.002, (200, 5))
    both = noise + np.array([0.02, 0.0, 0.01, 0.01, -0.02])
    assert wr.decide(both, rng) == {"state": True, "history": True, "form": "drift"}
    one_sided = noise + np.array([0.0, 0.0, 0.01, 0.0, 0.02])
    assert wr.decide(one_sided, rng) == {"state": False, "history": False, "form": "switching"}
    assert wr.decide(noise + 0.0009, rng)["state"] is False  # below the smallest difference of interest
    chance = wr.cohort_probabilities(both, rng, size=71, cohorts=50)
    assert chance == {"state": 1.0, "history": 1.0, "form_switching": 0.0, "form_drift": 1.0}
