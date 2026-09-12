"""Exploratory session-held-out comparisons of IBL choice-history models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

__all__ = ["AuditConfig", "prepare_trials", "design_matrix", "fit_logistic", "audit_history"]

CONTRASTS = (-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0)
PRIORS = (0.2, 0.5, 0.8)
MODELS = ("stimulus_block", "outcome_history", "evidence_history")


@dataclass(slots=True)
class AuditConfig:
    """Fixed analysis settings; this is an exploratory comparison, not a tuning loop."""

    seed: int = 20260905
    folds: int = 5
    weak_threshold: float = 0.125
    min_failures_per_bin: int = 10
    l2: float = 1.0

    def __post_init__(self) -> None:
        if self.folds < 2 or self.min_failures_per_bin < 1:
            raise ValueError("Require at least two folds and one failure per bin.")
        if not np.isfinite(self.l2) or self.l2 <= 0:
            raise ValueError("l2 must be finite and positive.")
        if not np.isfinite(self.weak_threshold) or not 0 <= self.weak_threshold < 1:
            raise ValueError("weak_threshold must be in [0, 1).")


def prepare_trials(df: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Build lagged covariates before exclusions; never bridge omissions or sessions."""
    data = df[df["task"] == "ibl_2afc"].copy()
    if data.empty or data.duplicated(["session_id", "trial_index"]).any():
        raise ValueError("Require nonempty IBL trials with unique session/trial keys.")
    data = data.sort_values(["session_id", "trial_index"]).reset_index(drop=True)
    groups = data.groupby("session_id", sort=False)
    data["previous_action"] = groups["action"].shift()
    data["previous_correct"] = groups["correct"].shift()
    data["previous_strength"] = groups["stimulus_contrast"].shift().abs()
    adjacent = data["trial_index"] - groups["trial_index"].shift() == 1
    data["prior_right"] = data["block_prior"].map(lambda value: value["p_right"])
    finite = np.isfinite(data[["stimulus_contrast", "prior_right", "previous_strength"]]).all(axis=1)
    keep = (
        adjacent & finite & data["action"].isin(["left", "right"])
        & data["previous_action"].isin(["left", "right"])
        & data["previous_correct"].notna()
    )
    data = data.loc[keep].copy()
    if data.empty:
        raise ValueError("No adjacent committed trials remain.")
    # Public priors include 1 - 0.8 = 0.19999999999999996. Canonicalize
    # numerical roundoff only, rejecting actual protocol differences.
    for column, levels in (("stimulus_contrast", CONTRASTS), ("prior_right", PRIORS),
                           ("previous_strength", sorted(set(abs(c) for c in CONTRASTS)))):
        values = data[column].to_numpy(float)
        matches = np.isclose(values[:, None], np.array(levels)[None, :], atol=1e-8, rtol=0)
        if not (matches.sum(axis=1) == 1).all():
            raise ValueError(f"Unrecognized IBL protocol values in {column}.")
        data[column] = np.array(levels)[matches.argmax(axis=1)]
    data["choice_right"] = (data.action == "right").astype(float)
    data["previous_right_signed"] = np.where(data.previous_action == "right", 1.0, -1.0)
    data["previous_correct"] = data.previous_correct.astype(float)
    data["previous_weak"] = (data.previous_strength <= config.weak_threshold).astype(float)
    data["retry"] = (data.action == data.previous_action).astype(float)
    return data.reset_index(drop=True)


def design_matrix(data: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str]]:
    """Use categorical stimulus-by-block controls and nested outcome-history terms."""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    columns = [np.ones(len(data))]
    names = ["intercept"]
    # Saturated task cells avoid imposing a shared psychometric slope across blocks.
    # Zero contrast in the neutral block is the intercept/reference cell.
    for contrast in CONTRASTS:
        for prior in PRIORS:
            if contrast == 0 and prior == 0.5:
                continue
            columns.append(((data.stimulus_contrast == contrast) & (data.prior_right == prior)).to_numpy(float))
            names.append(f"stimulus_{contrast}_prior_{prior}")
    if model != "stimulus_block":
        success = data.previous_correct.to_numpy(float)
        previous = data.previous_right_signed.to_numpy(float)
        columns.extend([success, previous * success, previous * (1 - success)])
        names.extend(["previous_success", "rewarded_choice", "unrewarded_choice"])
        for strength in (0.0625, 0.125, 0.25, 1.0):
            level = (data.previous_strength == strength).to_numpy(float)
            columns.extend([level, level * success])
            names.extend([f"previous_strength_{strength}", f"previous_strength_{strength}_success"])
        if model == "evidence_history":
            weak = data.previous_weak.to_numpy(float)
            columns.extend([previous * success * weak, previous * (1 - success) * weak])
            names.extend(["rewarded_choice_weak", "unrewarded_choice_weak"])
    return np.column_stack(columns), names


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Fit ridge logistic regression with an unpenalized intercept and checked convergence."""
    if len(np.unique(y)) != 2:
        raise ValueError("Training fold requires both choices.")
    penalty = np.ones(x.shape[1]) * l2
    penalty[0] = 0

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ beta
        loss = np.logaddexp(0.0, logits).sum() - y @ logits + 0.5 * np.sum(penalty * beta**2)
        gradient = x.T @ (expit(logits) - y) + penalty * beta
        return float(loss), gradient

    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Logistic fit did not converge: {fit.message}")
    return fit.x


def _gap(data: pd.DataFrame) -> dict[str, float | int | None]:
    failures = data[data.previous_correct == 0]
    weak = failures[failures.previous_weak == 1].retry
    strong = failures[failures.previous_weak == 0].retry
    return {
        "weak_count": len(weak), "strong_count": len(strong),
        "weak_retry": float(weak.mean()) if len(weak) else None,
        "strong_retry": float(strong.mean()) if len(strong) else None,
        "gap": float(weak.mean() - strong.mean()) if len(weak) and len(strong) else None,
    }


def audit_history(
    df: pd.DataFrame, config: AuditConfig, *, session_subjects: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return raw contrasts, explicit fold membership, and out-of-fold predictive losses."""
    data = prepare_trials(df, config)
    sessions = np.array(sorted(data.session_id.unique()))
    if len(sessions) < config.folds:
        raise ValueError("Fewer eligible sessions than folds.")
    if session_subjects is None:
        np.random.default_rng(config.seed).shuffle(sessions)
        folds = [list(map(str, part)) for part in np.array_split(sessions, config.folds)]
    else:
        if any(sid not in session_subjects or not session_subjects[sid] for sid in sessions):
            raise ValueError("Missing subject identity for an eligible session.")
        subjects = np.array(sorted({session_subjects[sid] for sid in sessions}))
        if len(subjects) < config.folds:
            raise ValueError("Fewer subjects than folds.")
        np.random.default_rng(config.seed).shuffle(subjects)
        folds = [[str(sid) for sid in sessions if session_subjects[sid] in part]
                 for part in np.array_split(subjects, config.folds)]
    losses: dict[str, np.ndarray] = {}
    coefficients: dict[str, list[dict[str, object]]] = {}
    for model in MODELS:
        x, names = design_matrix(data, model)
        y = data.choice_right.to_numpy(float)
        loss = np.full(len(data), np.nan)
        coefficients[model] = []
        for fold_index, held_out in enumerate(folds):
            test = data.session_id.isin(held_out).to_numpy()
            beta = fit_logistic(x[~test], y[~test], config.l2)
            logits = x[test] @ beta
            loss[test] = np.logaddexp(0, logits) - y[test] * logits
            coefficients[model].append({"fold": fold_index, "coefficients": dict(zip(names, map(float, beta)))})
        if not np.isfinite(loss).all():
            raise RuntimeError("Incomplete out-of-fold predictions.")
        losses[model] = loss
    per_session = []
    for session_id, group in data.groupby("session_id", sort=True):
        entry = {"session_id": str(session_id), "n_trials": len(group), **_gap(group)}
        entry["nll"] = {model: float(loss[group.index].mean()) for model, loss in losses.items()}
        per_session.append(entry)
    qualified = [entry["gap"] for entry in per_session
                 if entry["weak_count"] >= config.min_failures_per_bin
                 and entry["strong_count"] >= config.min_failures_per_bin]
    comparisons = {}
    for base, extended in zip(MODELS[:-1], MODELS[1:]):
        deltas = np.array([entry["nll"][base] - entry["nll"][extended] for entry in per_session])
        comparisons[f"{extended}_over_{base}"] = {
            "pooled_nll_gain": float(np.mean(losses[base] - losses[extended])),
            "equal_session_nll_gain": float(deltas.mean()),
            "median_session_nll_gain": float(np.median(deltas)),
            "sessions_improved": int(np.sum(deltas > 0)),
            "n_sessions": len(deltas),
            "fold_nll_gains": [float(np.mean((losses[base] - losses[extended])[data.session_id.isin(fold)]))
                               for fold in folds],
        }
    if session_subjects is not None:
        for entry in per_session:
            entry["subject"] = session_subjects[entry["session_id"]]
    result = {
        "status": ("exploratory_subject_held_out_previously_inspected_reference" if session_subjects
                   else "exploratory_session_held_out_not_animal_held_out"),
        "input_trials": len(df), "eligible_trials": len(data), "eligible_sessions": len(sessions),
        "pooled_retry": _gap(data),
        "qualified_session_retry": {
            "n": len(qualified),
            "median": float(np.median(qualified)) if qualified else None,
            "iqr": list(map(float, np.quantile(qualified, [0.25, 0.75]))) if qualified else None,
            "positive_fraction": float(np.mean(np.array(qualified) > 0)) if qualified else None,
        },
        "folds": folds, "comparisons": comparisons,
        "pooled_nll": {model: float(loss.mean()) for model, loss in losses.items()},
        "per_session": per_session, "fold_coefficients": coefficients,
        "limitations": [
            "All reference data were previously inspected; this is not a confirmatory holdout.",
            "Subject/lab identities are absent; sessions may share animals. No population p-values or confidence intervals.",
            "Observed past animal choices condition predictions; these are not autonomous agent rollouts.",
            "Block identity is an analyst control, not an observable cue supplied to an agent.",
            "Contrast is an evidence-strength proxy, not a measured subjective uncertainty.",
            "No causal mechanism, neural necessity, RT prediction, or novelty claim follows from this audit.",
        ],
    }

    if session_subjects is not None:
        result["session_subjects"] = {str(sid): session_subjects[sid] for sid in sessions}
        result["eligible_subjects"] = len(set(result["session_subjects"].values()))
        result["limitations"][1] = "Subjects are disjoint across folds; this previously inspected cohort is still exploratory."
        per_subject = []
        for subject in sorted(set(session_subjects[sid] for sid in sessions)):
            selected = [entry for entry in per_session if entry["subject"] == subject]
            count = sum(entry["n_trials"] for entry in selected)
            per_subject.append({
                "subject": subject, "n_trials": count, "n_sessions": len(selected),
                "nll": {model: sum(entry["nll"][model] * entry["n_trials"] for entry in selected) / count
                        for model in MODELS},
            })
        result["per_subject"] = per_subject
        for base, extended in zip(MODELS[:-1], MODELS[1:]):
            deltas = np.array([entry["nll"][base] - entry["nll"][extended] for entry in per_subject])
            result["comparisons"][f"{extended}_over_{base}"].update({
                "equal_subject_nll_gain": float(deltas.mean()),
                "subjects_improved": int(np.sum(deltas > 0)), "n_subjects": len(deltas),
            })
    return result
