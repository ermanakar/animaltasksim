"""Causal likelihood comparison of logistic history, drift, and switching models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        """Use the identical reference implementation without optional acceleration."""
        return lambda function: function
from scipy.optimize import minimize
from scipy.special import expit, softmax, ndtr

from eval.bias_history import task_design

__all__ = ["Sequence", "Fit", "prepare", "fit_model", "predict", "forward_backward", "MODELS"]
MODELS = ("task", "history", "drift", "drift_history", "switching", "switching_history")


@dataclass(slots=True)
class Sequence:
    """All committed observations, including unscored warm-up observations."""

    x: np.ndarray
    h: np.ndarray
    y: np.ndarray
    reset: np.ndarray
    score: np.ndarray
    rows: pd.DataFrame


@dataclass(slots=True)
class Fit:
    """Serializable model parameters and optimization diagnostics."""

    model: str
    beta: np.ndarray
    transition: np.ndarray
    offsets: np.ndarray
    objective: float
    converged: bool
    iterations: int
    gradient_max: float
    starts: list[dict]


def prepare(raw: pd.DataFrame, prefix: int = 100) -> Sequence:
    """Reset histories and hidden states at sessions, gaps, or omitted choices."""
    data = raw.sort_values(["session_id", "trial_index"]).reset_index(drop=True).copy()
    if data.empty or data.duplicated(["session_id", "trial_index"]).any():
        raise ValueError("Require nonempty unique trial keys")
    x = task_design(data)[0]
    h = np.zeros((len(data), 12))
    reset = np.zeros(len(data), dtype=np.bool_)
    score = np.zeros(len(data), dtype=np.bool_)
    valid = data.action.isin(["left", "right"]).to_numpy()
    for _, group in data.groupby("session_id", sort=False):
        history: list[tuple[float, float, float]] = []
        previous = None
        eligible = valid[group.index.to_numpy()[:prefix]].sum() >= min(80, prefix)
        for ordinal, i in enumerate(group.index):
            trial = data.at[i, "trial_index"]
            if previous is None or trial != previous + 1:
                history = []
            reset[i] = not history
            for lag, (a, c, weak) in enumerate(reversed(history[-5:])):
                h[i, 2 * lag:2 * lag + 2] = (a*c, a*(1-c))
            if history:
                a, c, weak = history[-1]
                h[i, 10:] = (a*c*weak, a*(1-c)*weak)
            score[i] = eligible and ordinal >= prefix and len(history) >= 5 and valid[i]
            if valid[i]:
                c = data.at[i, "correct"]
                if c not in (True, False, 0, 1):
                    raise ValueError("Committed response without binary correctness")
                history.append((1. if data.at[i, "action"] == "right" else -1., float(c),
                                float(abs(data.at[i, "stimulus_contrast"]) <= .125)))
            else:
                history = []
            previous = trial
    if not score.any():
        raise ValueError("No eligible scored observations")
    return Sequence(np.ascontiguousarray(x[valid]), np.ascontiguousarray(h[valid]),
                    (data.loc[valid, "action"] == "right").to_numpy(float),
                    reset[valid], score[valid], data.loc[valid].reset_index(drop=True))


@njit(cache=True)
def forward_backward(emission: np.ndarray, transition: np.ndarray, reset: np.ndarray) -> tuple:
    """Scaled exact HMM likelihood, posterior occupancy, and transition counts."""
    n, k = emission.shape
    alpha = np.zeros((n, k))
    prior = np.zeros((n, k))
    scale = np.zeros(n)
    for t in range(n):
        if reset[t] or t == 0:
            prior[t] = 1. / k
        else:
            for j in range(k):
                for i in range(k):
                    prior[t, j] += alpha[t-1, i] * transition[i, j]
        alpha[t] = prior[t] * emission[t]
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]
    gamma = np.zeros((n, k))
    counts = np.zeros((k, k))
    backward = np.ones(k)
    for t in range(n-1, -1, -1):
        gamma[t] = alpha[t] * backward
        gamma[t] /= gamma[t].sum()
        if t > 0 and not reset[t]:
            new_backward = np.zeros(k)
            for i in range(k):
                for j in range(k):
                    value = transition[i, j] * emission[t, j] * backward[j] / scale[t]
                    counts[i, j] += alpha[t-1, i] * value
                    new_backward[i] += value
            backward = new_backward
        else:
            backward = np.ones(k)
    return -np.log(scale).sum(), gamma, counts, prior


def _design(seq: Sequence, model: str) -> np.ndarray:
    return np.column_stack((seq.x, seq.h)) if "history" in model else seq.x


def _drift(grid: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.linspace(-3., 3., grid)
    mid = (offsets[:-1] + offsets[1:]) / 2
    edges = np.r_[-np.inf, mid, np.inf]
    # A discretized stationary AR(1), not PsyTrack's fitted random-walk weights.
    transition = np.diff(ndtr((edges[None, :] - .98*offsets[:, None])/.20), axis=1)
    return offsets, transition


def _objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray, reset: np.ndarray,
               model: str, offsets: np.ndarray, fixed: np.ndarray) -> tuple[float, np.ndarray]:
    k, p = len(offsets), x.shape[1]
    switching = model.startswith("switching")
    rows = k if switching else 1
    beta = theta[:rows*p].reshape(rows, p)
    a = softmax(theta[rows*p:].reshape(k, k), axis=1) if switching else fixed
    z = x @ beta.T + offsets
    probs = np.clip(expit(z), 1e-12, 1-1e-12)
    emission = np.where(y[:, None] == 1, probs, 1-probs)
    loss, gamma, counts, _ = forward_backward(emission, a, reset)
    residual = gamma * (probs-y[:, None])
    grad_beta = residual.T @ x if switching else residual.sum(1)[None, :] @ x
    # Ridge one per coefficient; weak Dirichlet MAP stabilization for transitions.
    loss += .5*np.sum(beta*beta)
    grad_beta += beta
    gradient = grad_beta.ravel()
    if switching:
        pseudo = .1*np.ones((k, k))
        loss -= np.sum(pseudo*np.log(a))
        count = counts+pseudo
        gradient = np.r_[gradient, (a*count.sum(1)[:, None]-count).ravel()]
    return float(loss), gradient


def fit_model(seq: Sequence, model: str, maxiter: int = 150, grid: int = 21) -> Fit:
    """Fit observed-data likelihood on training animals only; select training objective."""
    if model not in MODELS:
        raise ValueError("Unknown model")
    x = _design(seq, model)
    switching = model.startswith("switching")
    offsets, a = _drift(grid) if model.startswith("drift") else (np.zeros(3 if switching else 1), None)
    k, p = len(offsets), x.shape[1]
    if a is None:
        a = .97*np.eye(k)+.03*np.ones((k,k))/k
    starts, fitted = [], []
    # Warm-start from a fitted single-state GLM, as in established GLM-HMM workflows.
    # The initialization uses training responses only and never selects a test score.
    initial = np.zeros(p)
    if model != "task":
        initial_fit = minimize(_objective, initial,
                               args=(x, seq.y, seq.reset, "task", np.zeros(1), np.ones((1,1))),
                               method="L-BFGS-B", jac=True,
                               options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-4})
        if not initial_fit.success:
            raise RuntimeError("Single-state initialization did not converge")
        initial = initial_fit.x
    # Deterministic asymmetric starts avoid the identical-state stationary point.
    for strength in ((.5, 1.5) if switching else (0.,)):
        beta = np.tile(initial, (k if switching else 1, 1))
        if switching:
            beta[:, 0] += np.linspace(-strength, strength, k)
        theta = beta.ravel()
        if switching:
            theta = np.r_[theta, np.log(a).ravel()]
        result = minimize(_objective, theta, args=(x, seq.y, seq.reset, model, offsets, a),
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-4, "maxls": 30})
        starts.append({"objective": float(result.fun), "success": bool(result.success),
                       "iterations": int(result.nit), "message": str(result.message),
                       "gradient_max": float(np.max(np.abs(result.jac)))})
        fitted.append(result)
    result = min(fitted, key=lambda r: r.fun)
    rows = k if switching else 1
    beta = result.x[:rows*p].reshape(rows, p)
    if switching:
        a = softmax(result.x[rows*p:].reshape(k,k), axis=1)
    return Fit(model, beta, a, offsets, float(result.fun), bool(result.success),
               int(result.nit), float(np.max(np.abs(result.jac))), starts)


def predict(seq: Sequence, fit: Fit) -> np.ndarray:
    """Predict before observing each choice; backward states never enter predictions."""
    prob = np.clip(expit(_design(seq, fit.model) @ fit.beta.T + fit.offsets), 1e-12, 1-1e-12)
    emission = np.where(seq.y[:, None] == 1, prob, 1-prob)
    prior = forward_backward(emission, fit.transition, seq.reset)[3]
    return (prior * prob).sum(1)
