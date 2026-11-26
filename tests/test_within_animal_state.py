"""The latent-bias estimate is causal, and the four statistics and their reading behave as the protocol says."""
from __future__ import annotations

import numpy as np

from eval import within_animal as wa
from eval import within_animal_state as ws
from scripts import predict_from_state as script


def _schedule(seed: int, sessions: int = 8, trials: int = 500) -> wa.Sessions:
    rng = np.random.default_rng(seed)
    prior = np.r_[np.full(90, 0.5), np.repeat(np.tile([0.2, 0.8], 20), 41)][:trials]
    return wa.pad([{"contrast": rng.choice(wa.CONTRASTS, trials), "prior": prior, "choice": np.ones(trials),
                    "correct": np.zeros(trials)} for _ in range(sessions)])


def test_state_estimate_uses_only_earlier_choices() -> None:
    data, _ = ws.simulate_with_state(_schedule(0), np.random.default_rng(1), effect=False)
    before = ws.estimate_state(data, maxiter=40)
    altered = wa.Sessions(data.contrast, data.prior, data.choice.copy(), data.correct, data.length)
    altered.choice[0, 300:] *= -1  # session 0 is held out when the model is fitted on odd-numbered sessions
    after = ws.estimate_state(altered, maxiter=40)
    # The fit on even sessions changes, so compare the half whose parameters come from the untouched odd sessions.
    assert np.allclose(before["bias"][0, :301], after["bias"][0, :301])
    assert not np.allclose(before["bias"][0, 301:], after["bias"][0, 301:])
    assert np.allclose(before["bias"][:, 0], 0.0, atol=1e-12)  # no information before the first choice


def test_statistics_recover_planted_response_time_and_block_structure() -> None:
    rng = np.random.default_rng(2)
    schedule = _schedule(3, sessions=10, trials=600)
    live = np.ones(schedule.contrast.shape, dtype=bool)
    choice = np.where(rng.random(live.shape) < 0.5, 1.0, -1.0)
    choice[rng.random(live.shape) < 0.01] = 0.0
    bias = np.cumsum(rng.normal(0, 0.05, live.shape), axis=1)
    bias -= 0.3 * np.sign(schedule.prior - 0.5) * (np.arange(600)[None, :] % 41 < 10)  # crude lag, only for the sign test
    state = {"bias": bias, "history": rng.normal(0, 0.3, live.shape), "block": 0.4 * np.sign(schedule.prior - 0.5)}
    rt = np.exp(np.log(300) - 0.1 * choice * bias + 0.2 * np.abs(bias) + rng.normal(0, 0.3, live.shape))
    data = wa.Sessions(schedule.contrast, schedule.prior, choice, (rng.random(live.shape) < 0.8) * (choice != 0), schedule.length)
    stats = ws.animal_statistics(data, rt, state)
    assert abs(stats["p1_congruence"] + 0.1) < 0.02 and abs(stats["p2_magnitude"] - 0.2) < 0.03
    assert stats["rt_trials"] == int((choice != 0).sum()) and stats["block_changes"] > 50
    few = ws.animal_statistics(data, np.where(np.arange(600)[None, :] < 40, rt, np.nan), state)
    assert few["p1_congruence"] is None and few["rt_trials"] == 0  # under half of each session's choices are timed


def test_omission_statistic_needs_enough_events_and_follows_the_lean() -> None:
    rng = np.random.default_rng(4)
    schedule = _schedule(5, sessions=10, trials=600)
    bias = 0.8 * np.sin(np.arange(600) / 60)[None, :] * rng.normal(1, 0.2, (10, 1))
    omit = rng.random(bias.shape) < 0.01 * np.exp(2 * np.abs(bias))
    data = wa.Sessions(schedule.contrast, schedule.prior, np.where(omit, 0.0, 1.0), np.zeros(bias.shape), schedule.length)
    state = {"bias": bias, "history": np.zeros(bias.shape), "block": np.zeros(bias.shape)}
    stats = ws.animal_statistics(data, np.full(bias.shape, np.nan), state)
    assert stats["omissions"] >= 10 and stats["p3_omission"] > 0.05
    rare = wa.Sessions(schedule.contrast, schedule.prior, np.where(omit & (rng.random(bias.shape) < 0.02), 0.0, 1.0),
                       np.zeros(bias.shape), schedule.length)
    assert ws.animal_statistics(rare, np.full(bias.shape, np.nan), state)["p3_omission"] is None


def test_reading_and_gate() -> None:
    rng = np.random.default_rng(6)
    noise = rng.normal(0, 0.01, 71)
    assert ws.verdict(noise - 0.06, "p1_congruence", rng, 500)["reading"] == "supported"
    assert ws.verdict(noise + 0.06, "p1_congruence", rng, 500)["reading"] == "opposite"
    assert ws.verdict(noise + 0.015, "p2_magnitude", rng, 500)["reading"].startswith("not supported")  # real but too small
    quiet = {key: {"reading": "not supported at the size of interest"} for key in ws.SIZES}
    found = quiet | {key: {"reading": "supported"} for key in ("p1_congruence", "p2_magnitude", "p3_omission")}
    assert script.check_gate({"null": quiet, "effect": found})
    assert not script.check_gate({"null": found, "effect": found})
    assert not script.check_gate({"null": quiet, "effect": quiet})
