"""Causal estimates of the latent side bias, and what they predict about response time, omissions and block changes."""

from __future__ import annotations

import numpy as np

from eval.within_animal import CONTRASTS, LAGS, WEAK, Sessions, _emission, _predictors, design, fit, hmm
from eval.within_animal_recovery import DRIFT_SD, RHO, _subset, history_weights

__all__ = ["MODEL", "SIZES", "DIRECTIONS", "estimate_state", "animal_statistics", "simulate_with_state", "verdict"]

MODEL = "drift_history"
SIZES = {"p1_congruence": 0.02, "p2_magnitude": 0.02, "p3_omission": 0.02, "d1_block_lag": 0.05}
DIRECTIONS = {"p1_congruence": -1, "p2_magnitude": 1, "p3_omission": 1, "d1_block_lag": -1}
BLOCK = len(CONTRASTS)  # column of the block term in the task predictors
BIN, LAST_BIN, RUN = 100, 9, 5
MIN_RT_TRIALS, MIN_OMISSIONS = 500, 10
RT_BOUNDS = (50.0, 60000.0)


def estimate_state(data: Sessions, maxiter: int = 5000) -> dict[str, np.ndarray]:
    """Expected side bias before each choice, from parameters fitted on the other half of the sessions.

    Arrays are [session, trial]. `bias` uses only choices on earlier trials of the same session.
    `history` and `block` are the fitted model's log-odds contributions for a right choice.
    """
    full, count = design(data), len(data.length)
    shape = data.contrast.shape
    out = {key: np.zeros(shape) for key in ("bias", "history", "block")}
    converged = True
    for train, test in ((np.arange(0, count, 2), np.arange(1, count, 2)), (np.arange(1, count, 2), np.arange(0, count, 2))):
        fitted = fit(_subset(full, train, data.length), MODEL, maxiter=maxiter)
        converged &= fitted.converged
        held = _subset(full, test, data.length)
        x, beta = _predictors(held, MODEL), fitted.beta[0]
        _, emission = _emission((x @ beta)[:, :, None] + fitted.offsets, held)
        predicted = hmm(emission, fitted.transition, fitted.initial, held.live, backward=False)[3]
        last = held.live.shape[0]
        out["bias"][test, :last] = ((predicted @ fitted.offsets) * held.live).T
        out["history"][test, :last] = (held.h @ beta[BLOCK + 1:]).T
        out["block"][test, :last] = (held.x[:, :, BLOCK] * beta[BLOCK]).T
    return out | {"converged": np.array(converged)}


def _centre(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    sums = np.zeros((groups.max() + 1, values.shape[1]))
    np.add.at(sums, groups, values)
    return values - (sums / np.bincount(groups)[:, None])[groups]


def animal_statistics(data: Sessions, rt: np.ndarray, state: dict[str, np.ndarray]) -> dict[str, float | int | None]:
    """The four per-animal quantities of the protocol; a quantity is None when the animal has too little data for it."""
    sessions, trials = data.contrast.shape
    index = np.arange(trials)[None, :]
    live = index < data.length[:, None]
    committed = live & (data.choice != 0)
    bias, size = state["bias"], np.abs(state["bias"])
    position = np.minimum(index // BIN, LAST_BIN) * np.ones((sessions, 1), dtype=int)
    session = np.arange(sessions)[:, None] * np.ones((1, trials), dtype=int)
    out: dict[str, float | int | None] = {key: None for key in SIZES}

    timed = committed & np.isfinite(rt) & (rt > 0)
    covered = timed.sum(1) >= 0.5 * np.maximum(committed.sum(1), 1)
    rows = timed & covered[:, None]
    out["rt_trials"] = int(rows.sum())
    if rows.sum() >= MIN_RT_TRIALS:
        choice, strength = data.choice[rows], np.abs(data.contrast[rows])
        columns = [choice * bias[rows], size[rows], choice, data.correct[rows], choice * state["history"][rows],
                   choice * state["block"][rows]]
        columns += [(strength == level).astype(float) for level in sorted(set(np.abs(CONTRASTS)) - {0.0})]
        columns += [(position[rows] == b).astype(float) for b in range(1, LAST_BIN + 1)]
        groups = np.unique(session[rows], return_inverse=True)[1]
        x = _centre(np.column_stack(columns), groups)
        y = _centre(np.log(np.clip(rt[rows], *RT_BOUNDS))[:, None], groups)[:, 0]
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        out["p1_congruence"], out["p2_magnitude"] = float(coefficients[0]), float(coefficients[1])

    run = np.zeros((sessions, trials), dtype=int)  # committed trials in a row before each trial
    for t in range(1, trials):
        run[:, t] = np.where(committed[:, t - 1], run[:, t - 1] + 1, 0)
    usable = live & (run >= RUN)
    omitted = (usable & (data.choice == 0))[usable]
    out["omissions"] = int(omitted.sum())
    if omitted.sum() >= MIN_OMISSIONS:
        groups = np.unique(session[usable], return_inverse=True)[1]
        bins = _centre(np.column_stack([(position[usable] == b).astype(float) for b in range(1, LAST_BIN + 1)]), groups)
        lean = _centre(size[usable][:, None], groups)[:, 0]
        residual = lean - bins @ np.linalg.lstsq(bins, lean, rcond=None)[0]  # lean beyond its session and position
        out["p3_omission"] = float(residual[omitted].mean() - residual[~omitted].mean())

    early, late = [], []
    for s in range(sessions):
        prior = data.prior[s, :data.length[s]]
        changes = np.flatnonzero((np.abs(np.diff(prior)) > 0.5)) + 1  # only 0.2 <-> 0.8
        for k, start in enumerate(changes):
            stop = changes[k + 1] if k + 1 < len(changes) else data.length[s]
            signed = bias[s, start:stop] * np.sign(prior[start] - 0.5)
            if len(signed) >= 10:
                early.append(signed[:10].mean())
            if len(signed) >= 50:
                late.append(signed[30:50].mean())
    out["block_changes"] = len(early)
    out["d1_block_lag"] = float(np.mean(early)) if early else None
    out["block_late"] = float(np.mean(late)) if late else None
    return out


def verdict(values: np.ndarray, key: str, rng: np.random.Generator, resamples: int = 10000) -> dict[str, object]:
    """Equal-animal mean, bootstrap interval, and the protocol's three-way reading against the size of interest."""
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(len(values), size=(resamples, len(values)))].mean(1)
    low, high = (float(v) for v in np.quantile(means, [0.025, 0.975]))
    size, direction = SIZES[key], DIRECTIONS[key]
    toward, away = (low > size, high < -size) if direction > 0 else (high < -size, low > size)
    reading = "supported" if toward else "opposite" if away else "not supported at the size of interest"
    return {"animals": len(values), "mean": float(values.mean()), "interval": [low, high],
            "animals_in_predicted_direction": int((values * direction > 0).sum()), "reading": reading}


def simulate_with_state(schedule: Sessions, rng: np.random.Generator, effect: bool) -> tuple[Sessions, np.ndarray]:
    """Drift-plus-history choices on a real schedule, with response times and omissions that do or do not follow the state."""
    sessions, trials = schedule.contrast.shape
    live = np.arange(trials)[None, :] < schedule.length[:, None]
    contrast = schedule.contrast
    sensitivity, constant = float(np.exp(rng.normal(0, 0.2))), float(rng.normal(0, 0.4))
    weights = history_weights(1.0) * float(np.exp(rng.normal(0, 0.4)))
    sd = DRIFT_SD * float(np.exp(rng.normal(0, 0.3)))
    sensory = sensitivity * 4 * np.sign(contrast) * np.sqrt(np.abs(contrast))
    block = 0.4 * np.sign(np.round(schedule.prior - 0.5, 6))
    side = np.where(contrast != 0, np.sign(contrast), np.where(rng.random(contrast.shape) < schedule.prior, 1.0, -1.0))
    choice, correct, rt = np.zeros_like(contrast), np.zeros_like(contrast), np.full(contrast.shape, np.nan)
    run, drift = np.zeros(sessions, dtype=int), rng.normal(0, sd, sessions)
    for t in range(trials):
        z = constant + block[:, t] + sensory[:, t] + drift
        for lag in range(1, min(LAGS, t) + 1):
            signed = choice[:, t - lag] * (run >= lag)
            z = z + signed * np.where(correct[:, t - lag] == 1, weights[2 * (lag - 1)], weights[2 * (lag - 1) + 1])
        if t:
            weak = choice[:, t - 1] * (run >= 1) * (np.abs(contrast[:, t - 1]) <= WEAK)
            z = z + weak * np.where(correct[:, t - 1] == 1, weights[2 * LAGS], weights[2 * LAGS + 1])
        omit = rng.random(sessions) < 1 / (1 + np.exp(-(np.log(0.004 / 0.996) + (1.0 * np.abs(drift) if effect else 0.0))))
        made = live[:, t] & ~omit
        chosen = np.where(rng.random(sessions) < 1 / (1 + np.exp(-z)), 1.0, -1.0)
        choice[:, t], correct[:, t] = chosen * made, (chosen == side[:, t]) * made
        log_rt = (np.log(400.0) - 0.5 * np.sqrt(np.abs(contrast[:, t])) + 0.04 * min(t // BIN, LAST_BIN) + 0.05 * chosen
                  - 0.10 * (chosen == side[:, t]) + rng.normal(0, 0.5, sessions))
        if effect:
            log_rt = log_rt - 0.06 * chosen * drift + 0.06 * np.abs(drift)
        rt[:, t] = np.where(made, np.exp(log_rt), np.nan)
        run = np.where(made, run + 1, 0)
        drift = RHO * drift + rng.normal(0, sd * np.sqrt(1 - RHO ** 2), sessions)
    return Sessions(contrast, schedule.prior, choice, correct, schedule.length), rt
