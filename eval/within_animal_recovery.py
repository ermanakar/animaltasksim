"""Synthetic choices on real schedules, and the cohort-level detection rules for within-animal models."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from eval.within_animal import LAGS, MODELS, WEAK, Design, Sessions, design, fit, predict

__all__ = ["CELLS", "CONTRASTS_TESTED", "DELTA", "history_weights", "simulate", "score_animal",
           "decide", "cohort_probabilities"]

DELTA = 0.001
# cell -> (latent process, history strength)
CELLS = {"fixed": (None, 0.0), "history": (None, 1.0), "drift": ("drift", 0.0), "switching": ("switching", 0.0),
         "drift_history": ("drift", 1.0), "switching_history": ("switching", 1.0),
         "drift_history_half": ("drift", 0.5), "switching_history_half": ("switching", 0.5)}
CONTRASTS_TESTED = {"state_drift": ("history", "drift_history"), "state_switching": ("history", "switching_history"),
                    "history_drift": ("drift", "drift_history"), "history_switching": ("switching", "switching_history"),
                    "form": ("drift_history", "switching_history")}
RHO, DRIFT_SD = 0.995, 0.6
STATE_GAIN, STATE_BIAS = np.array([1.0, 0.35, 0.35]), np.array([0.0, -1.5, 1.5])
STATE_TRANSITION = np.array([[0.98, 0.01, 0.01], [0.03, 0.965, 0.005], [0.03, 0.005, 0.965]])
STATE_START = np.array([0.6, 0.2, 0.2])  # stationary distribution of the matrix above


def history_weights(strength: float) -> np.ndarray:
    """Generating weights in the column order of `within_animal.history_features`."""
    weights = np.zeros(2 * LAGS + 2)
    for lag in range(LAGS):
        weights[2 * lag], weights[2 * lag + 1] = 0.30 * 0.6 ** lag, -0.25 * 0.6 ** lag
    weights[2 * LAGS], weights[2 * LAGS + 1] = 0.15, -0.30
    return strength * weights


def simulate(schedule: Sessions, cell: str, rng: np.random.Generator) -> tuple[Sessions, dict[str, float]]:
    """New choices for one animal; only its contrasts, priors and omission positions are read."""
    latent, strength = CELLS[cell]
    sessions, trials = schedule.contrast.shape
    live = np.arange(trials)[None, :] < schedule.length[:, None]
    omitted = live & (schedule.choice == 0)
    contrast = schedule.contrast
    sensitivity, bias = float(np.exp(rng.normal(0, 0.2))), float(rng.normal(0, 0.4))
    scale, drift_sd = float(np.exp(rng.normal(0, 0.4))), DRIFT_SD * float(np.exp(rng.normal(0, 0.3)))
    sensory = sensitivity * 4 * np.sign(contrast) * np.sqrt(np.abs(contrast))
    block = 0.4 * np.sign(np.round(schedule.prior - 0.5, 6))
    side = np.where(contrast != 0, np.sign(contrast), np.where(rng.random(contrast.shape) < schedule.prior, 1.0, -1.0))
    weights = history_weights(strength) * scale
    choice, correct = np.zeros_like(contrast), np.zeros_like(contrast)
    run = np.zeros(sessions, dtype=int)
    drift = rng.normal(0, drift_sd, sessions)
    state = rng.choice(3, sessions, p=STATE_START)
    cumulative = STATE_TRANSITION.cumsum(1)
    for t in range(trials):
        z = bias + block[:, t]
        if latent == "switching":
            z = z + STATE_GAIN[state] * sensory[:, t] + STATE_BIAS[state]
        else:
            z = z + sensory[:, t] + (drift if latent == "drift" else 0.0)
        if strength:
            for lag in range(1, min(LAGS, t) + 1):
                signed = choice[:, t - lag] * (run >= lag)
                z = z + signed * np.where(correct[:, t - lag] == 1, weights[2 * (lag - 1)], weights[2 * (lag - 1) + 1])
            if t:
                weak = choice[:, t - 1] * (run >= 1) * (np.abs(contrast[:, t - 1]) <= WEAK)
                z = z + weak * np.where(correct[:, t - 1] == 1, weights[2 * LAGS], weights[2 * LAGS + 1])
        made = live[:, t] & ~omitted[:, t]
        chosen = np.where(rng.random(sessions) < 1 / (1 + np.exp(-z)), 1.0, -1.0)
        choice[:, t] = chosen * made
        correct[:, t] = (chosen == side[:, t]) * made
        run = np.where(made, run + 1, 0)
        drift = RHO * drift + rng.normal(0, drift_sd * np.sqrt(1 - RHO ** 2), sessions)
        state = (rng.random(sessions)[:, None] > cumulative[state]).sum(1)
    truth = {"sensitivity": sensitivity, "bias": bias, "history_scale": scale * strength,
             "drift_sd": drift_sd if latent == "drift" else 0.0}
    return replace(schedule, choice=choice, correct=correct), truth


def _subset(data: Design, rows: np.ndarray, length: np.ndarray) -> Design:
    last = int(length[rows].max())
    return Design(*(np.ascontiguousarray(getattr(data, name)[:last, rows]) for name in ("x", "h", "y", "committed", "live")))


def score_animal(data: Sessions, states: int = 41, models: tuple[str, ...] = MODELS) -> dict[str, object]:
    """Fit on even sessions and score odd ones, then the reverse; mean held-out loss per committed trial."""
    full = design(data)
    halves = [np.arange(0, len(data.length), 2), np.arange(1, len(data.length), 2)]
    committed = int(full.committed.sum())
    out: dict[str, object] = {"trials": committed, "nll": {}, "converged": {}, "drift": {}}
    for model in models:
        total, converged, parameters = 0.0, [], []
        for train, test in (halves, halves[::-1]):
            fitted = fit(_subset(full, train, data.length), model, states=states)
            total += float(np.nansum(predict(_subset(full, test, data.length), fitted)))
            converged.append(fitted.converged)
            parameters.append(fitted.extra)
        out["nll"][model] = total / committed
        out["converged"][model] = bool(all(converged))
        if model.startswith("drift"):
            out["drift"][model] = {key: float(np.mean([p[key] for p in parameters])) for key in ("rho", "sd")}
    return out


def _bounds(gains: np.ndarray, rng: np.random.Generator, resamples: int) -> tuple[np.ndarray, np.ndarray]:
    means = gains[rng.integers(len(gains), size=(resamples, len(gains)))].mean(1)
    return np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def decide(gains: np.ndarray, rng: np.random.Generator, resamples: int = 500) -> dict[str, object]:
    """Apply the three detection rules to per-animal gains in `CONTRASTS_TESTED` column order."""
    low, high = _bounds(gains, rng, resamples)
    form = "switching" if low[4] > DELTA else "drift" if high[4] < -DELTA else "unresolved"
    return {"state": bool(low[0] > DELTA or low[1] > DELTA), "history": bool(low[2] > DELTA and low[3] > DELTA), "form": form}


def cohort_probabilities(gains: np.ndarray, rng: np.random.Generator, size: int = 71,
                         cohorts: int = 2000) -> dict[str, float]:
    """Chance that a cohort of independent animals triggers each rule, by resampling simulated animals."""
    tally = {"state": 0, "history": 0, "form_switching": 0, "form_drift": 0}
    for _ in range(cohorts):
        outcome = decide(gains[rng.integers(len(gains), size=size)], rng)
        tally["state"] += outcome["state"]
        tally["history"] += outcome["history"]
        if outcome["form"] != "unresolved":
            tally[f"form_{outcome['form']}"] += 1
    return {key: value / cohorts for key, value in tally.items()}
