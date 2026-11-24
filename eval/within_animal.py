"""Within-animal choice models: fixed, trial history, fitted slow drift, and discrete states.

Each model is fitted to some sessions of one animal and scored on that animal's other sessions.
Held-out predictions use only earlier trials of the same held-out session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr, softmax

__all__ = ["MODELS", "CONTRASTS", "Sessions", "Design", "Fit", "pad", "design", "hmm", "fit", "predict"]

MODELS = ("fixed", "history", "drift", "drift_history", "switching", "switching_history")
CONTRASTS = (-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0)
LAGS, WEAK, STATES = 5, 0.125, 3
RHO_BOUNDS, SD_BOUNDS = (0.90, 0.9995), (1e-3, 3.0)


@dataclass(slots=True)
class Sessions:
    """Padded trials of one animal; rows are sessions and `choice` is 0 for omissions and padding."""

    contrast: np.ndarray
    prior: np.ndarray
    choice: np.ndarray
    correct: np.ndarray
    length: np.ndarray


@dataclass(slots=True)
class Design:
    """Time-major arrays: task predictors, history predictors, right choices, committed and live masks."""

    x: np.ndarray
    h: np.ndarray
    y: np.ndarray
    committed: np.ndarray
    live: np.ndarray


@dataclass(slots=True)
class Fit:
    """Fitted parameters and optimiser diagnostics for one model."""

    model: str
    beta: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    offsets: np.ndarray
    objective: float
    converged: bool
    extra: dict[str, float]


def pad(sessions: list[dict[str, np.ndarray]]) -> Sessions:
    """Stack variable-length sessions; choice is +1 right, -1 left, 0 omitted."""
    length = np.array([len(s["choice"]) for s in sessions])
    out = {key: np.zeros((len(sessions), int(length.max()))) for key in ("contrast", "prior", "choice", "correct")}
    out["prior"] += 0.5
    for row, session in enumerate(sessions):
        for key, target in out.items():
            target[row, :length[row]] = session[key]
    return Sessions(length=length, **out)


def history_features(choice: np.ndarray, correct: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """Rewarded and unrewarded choices at lags 1-5 plus two weak-evidence terms; cleared by any omission."""
    sessions, trials = choice.shape
    committed = choice != 0
    run = np.zeros((sessions, trials), dtype=int)  # consecutive committed trials immediately before t
    for t in range(1, trials):
        run[:, t] = np.where(committed[:, t - 1], run[:, t - 1] + 1, 0)
    h = np.zeros((sessions, trials, 2 * LAGS + 2))
    for lag in range(1, LAGS + 1):
        valid = run[:, lag:] >= lag
        a, c = choice[:, :-lag], correct[:, :-lag]
        h[:, lag:, 2 * (lag - 1)] = a * c * valid
        h[:, lag:, 2 * (lag - 1) + 1] = a * (1 - c) * valid
    weak = (np.abs(contrast[:, :-1]) <= WEAK) * (run[:, 1:] >= 1)
    h[:, 1:, 2 * LAGS] = choice[:, :-1] * correct[:, :-1] * weak
    h[:, 1:, 2 * LAGS + 1] = choice[:, :-1] * (1 - correct[:, :-1]) * weak
    return h


def design(data: Sessions) -> Design:
    """One weight per signed contrast, one block term, and causal history terms."""
    levels = np.asarray(CONTRASTS)
    match = np.isclose(data.contrast[:, :, None], levels[None, None, :], atol=1e-8, rtol=0)
    live = np.arange(data.contrast.shape[1])[None, :] < data.length[:, None]
    if not match[live].any(axis=1).all():
        raise ValueError("Contrast outside the standard set")
    block = np.sign(np.round(data.prior - 0.5, 6))
    x = np.concatenate([match.astype(float), block[:, :, None]], axis=2) * live[:, :, None]
    h = history_features(data.choice, data.correct, data.contrast)
    swap = lambda a: np.ascontiguousarray(np.swapaxes(a, 0, 1))  # noqa: E731
    return Design(swap(x), swap(h), swap((data.choice == 1).astype(float)), swap(data.choice != 0), swap(live))


def hmm(emission: np.ndarray, transition: np.ndarray, initial: np.ndarray, live: np.ndarray,
        backward: bool = True) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Scaled forward-backward over sessions in parallel.

    `emission` is [trial, session, state] and equals 1 where no choice was observed. Returns the
    log likelihood, state posteriors, expected transition counts, and one-step-ahead state
    predictions made before each choice.
    """
    trials, sessions, states = emission.shape
    predicted, alpha = np.empty_like(emission), np.empty_like(emission)
    scale = np.empty((trials, sessions))
    predicted[0] = initial
    for t in range(trials):
        if t:
            predicted[t] = alpha[t - 1] @ transition
        joint = predicted[t] * emission[t]
        scale[t] = joint.sum(1)
        alpha[t] = joint / scale[t][:, None]
    loglik = float(np.log(scale).sum())
    gamma, counts = np.empty_like(emission), np.zeros((states, states))
    if not backward:
        return loglik, gamma, counts, predicted
    message = np.ones((sessions, states))
    gamma[-1] = alpha[-1]
    for t in range(trials - 1, 0, -1):
        weighted = emission[t] * message / scale[t][:, None]
        counts += alpha[t - 1].T @ (weighted * live[t][:, None])  # padding after a session adds nothing
        message = weighted @ transition.T
        gamma[t - 1] = alpha[t - 1] * message
    return loglik, gamma, counts * transition, predicted


def _grid(states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    units = np.linspace(-3.0, 3.0, states)
    edges = np.r_[-np.inf, (units[:-1] + units[1:]) / 2, np.inf]
    return units, edges, np.diff(ndtr(edges))


def _drift_transition(rho: float, units: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discretised unit-variance AR(1) and its derivative with respect to rho."""
    tau = np.sqrt(1 - rho * rho)
    q = (edges[None, :] - rho * units[:, None]) / tau
    finite = np.where(np.isfinite(edges), edges, 0.0)[None, :]
    slope = np.where(np.isfinite(q), np.exp(-0.5 * np.where(np.isfinite(q), q, 0.0) ** 2) / np.sqrt(2 * np.pi)
                     * (rho * finite - units[:, None]) / tau ** 3, 0.0)
    return np.diff(ndtr(q), axis=1), np.diff(slope, axis=1)


def _predictors(data: Design, model: str) -> np.ndarray:
    return np.concatenate([data.x, data.h], axis=2) if "history" in model else data.x


def _emission(z: np.ndarray, data: Design) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(expit(z), 1e-12, 1 - 1e-12)
    chosen = np.where(data.y[:, :, None] == 1, p, 1 - p)
    return p, np.where(data.committed[:, :, None], chosen, 1.0)


def _glm_loss(beta: np.ndarray, x: np.ndarray, data: Design) -> tuple[float, np.ndarray]:
    z = x @ beta
    loss = np.sum((np.logaddexp(0, z) - data.y * z) * data.committed) + 0.5 * beta @ beta
    return float(loss), np.einsum("ts,tsp->p", (expit(z) - data.y) * data.committed, x) + beta


def _drift_loss(theta: np.ndarray, x: np.ndarray, data: Design, grid: tuple) -> tuple[float, np.ndarray]:
    units, edges, initial = grid
    beta, rho, sd = theta[:-2], theta[-2], theta[-1]
    transition, slope = _drift_transition(rho, units, edges)
    p, emission = _emission((x @ beta)[:, :, None] + sd * units, data)
    loglik, gamma, counts, _ = hmm(emission, transition, initial, data.live)
    residual = gamma * (p - data.y[:, :, None]) * data.committed[:, :, None]
    ratio = np.divide(counts, transition, out=np.zeros_like(counts), where=transition > 1e-300)
    gradient = np.r_[np.einsum("ts,tsp->p", residual.sum(2), x) + beta, -np.sum(ratio * slope), np.sum(residual * units)]
    return float(-loglik + 0.5 * beta @ beta), gradient


def _switching_loss(theta: np.ndarray, x: np.ndarray, data: Design) -> tuple[float, np.ndarray]:
    width = x.shape[2]
    beta = theta[:STATES * width].reshape(STATES, width)
    transition = softmax(theta[STATES * width:].reshape(STATES, STATES), axis=1)
    p, emission = _emission(x @ beta.T, data)
    loglik, gamma, counts, _ = hmm(emission, transition, np.full(STATES, 1 / STATES), data.live)
    residual = gamma * (p - data.y[:, :, None]) * data.committed[:, :, None]
    pseudo = counts + 0.1  # weak symmetric Dirichlet prior on each transition row
    loss = -loglik + 0.5 * np.sum(beta * beta) - 0.1 * np.sum(np.log(transition))
    gradient = np.r_[(np.einsum("tsk,tsp->kp", residual, x) + beta).ravel(),
                     (transition * pseudo.sum(1)[:, None] - pseudo).ravel()]
    return float(loss), gradient


def fit(data: Design, model: str, maxiter: int = 500, states: int = 41) -> Fit:
    """Fit one model to one animal's training sessions by penalised likelihood."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model}")
    x = _predictors(data, model)
    width = x.shape[2]
    options = {"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-5}
    glm = minimize(_glm_loss, np.zeros(width), args=(x, data), jac=True, method="L-BFGS-B", options=options)
    if model in ("fixed", "history"):
        return Fit(model, glm.x[None, :], np.ones((1, 1)), np.ones(1), np.zeros(1), float(glm.fun), bool(glm.success), {})
    if model.startswith("drift"):
        grid = _grid(states)
        result = minimize(_drift_loss, np.r_[glm.x, 0.98, 0.5], args=(x, data, grid), jac=True, method="L-BFGS-B",
                          bounds=[(None, None)] * width + [RHO_BOUNDS, SD_BOUNDS], options=options)
        rho, sd = result.x[-2:]
        return Fit(model, result.x[None, :-2], _drift_transition(rho, grid[0], grid[1])[0], grid[2], sd * grid[0],
                   float(result.fun), bool(result.success), {"rho": float(rho), "sd": float(sd)})
    # Deterministic asymmetric starts: one engaged state and two flatter, oppositely biased states.
    sticky = np.log(0.97 * np.eye(STATES) + 0.01)
    results = []
    for strength in (0.5, 1.5):
        beta = np.tile(glm.x, (STATES, 1))
        beta[1:, :len(CONTRASTS)] *= 0.5
        beta[1, :len(CONTRASTS)] -= strength
        beta[2, :len(CONTRASTS)] += strength
        results.append(minimize(_switching_loss, np.r_[beta.ravel(), sticky.ravel()], args=(x, data), jac=True,
                                method="L-BFGS-B", options=options | {"maxls": 30}))
    result = min(results, key=lambda r: r.fun)
    return Fit(model, result.x[:STATES * width].reshape(STATES, width),
               softmax(result.x[STATES * width:].reshape(STATES, STATES), axis=1), np.full(STATES, 1 / STATES),
               np.zeros(STATES), float(result.fun), bool(result.success),
               {"start_objectives": [float(r.fun) for r in results]})


def predict(data: Design, fitted: Fit) -> np.ndarray:
    """Negative log likelihood of each committed held-out choice, predicted before it is observed."""
    p, emission = _emission(_predictors(data, fitted.model) @ fitted.beta.T + fitted.offsets, data)
    predicted = hmm(emission, fitted.transition, fitted.initial, data.live, backward=False)[3]
    right = np.clip((predicted * p).sum(2), 1e-12, 1 - 1e-12)
    return np.where(data.committed, -np.log(np.where(data.y == 1, right, 1 - right)), np.nan)
