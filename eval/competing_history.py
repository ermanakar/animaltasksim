"""Development comparisons of evidence history, longer memory, and leaky learning."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eval.history_audit import AuditConfig, design_matrix, fit_logistic, prepare_trials

__all__ = ["ComparisonConfig", "prepare_comparison", "comparison_design", "make_folds", "compare"]
MODELS = ("outcome_history", "evidence_history", "long_history", "long_evidence",
          "learning_history", "learning_evidence", "combined_history", "combined_evidence")
PAIRS = (("outcome_history", "evidence_history"), ("long_history", "long_evidence"),
         ("learning_history", "learning_evidence"), ("combined_history", "combined_evidence"))


@dataclass(slots=True)
class ComparisonConfig:
    """Fixed exploratory controls; no validation-based hyperparameter selection."""

    seed: int = 20260912
    folds: int = 5
    lags: int = 5
    rates: tuple[float, ...] = (0.05, 0.2)
    l2: float = 1.0

    def __post_init__(self) -> None:
        if self.folds < 2 or self.lags < 2 or not np.isfinite(self.l2) or self.l2 <= 0:
            raise ValueError("Require multiple folds/lags and positive finite ridge penalty")
        if not self.rates or len(set(self.rates)) != len(self.rates) or any(not 0 < r < 1 for r in self.rates):
            raise ValueError("Learning rates must be distinct and lie strictly between zero and one")


def prepare_comparison(raw: pd.DataFrame, config: ComparisonConfig) -> pd.DataFrame:
    """Use identical contiguous histories for every model; states precede current outcome."""
    data = raw[raw.task == "ibl_2afc"].sort_values(["session_id", "trial_index"]).reset_index(drop=True).copy()
    if data.empty or data.duplicated(["session_id", "trial_index"]).any():
        raise ValueError("Require nonempty unique IBL trial keys")
    if not np.isfinite(data.stimulus_contrast.to_numpy(float)).all():
        raise ValueError("Nonfinite stimulus")
    committed = data.action.isin(["left", "right"])
    if not data.loc[committed, "correct"].isin([True, False, 0, 1]).all():
        raise ValueError("Committed trials need binary correctness")
    grouped = data.groupby("session_id", sort=False)
    keep = committed.copy()
    for lag in range(1, config.lags + 1):
        action = grouped.action.shift(lag)
        correct = grouped.correct.shift(lag)
        contiguous = data.trial_index - grouped.trial_index.shift(lag) == lag
        keep &= contiguous & action.isin(["left", "right"]) & correct.notna()
        signed = np.where(action == "right", 1.0, -1.0)
        data[f"rewarded_lag_{lag}"] = signed * (correct == True).to_numpy(float)  # noqa: E712
        data[f"unrewarded_lag_{lag}"] = signed * (correct == False).to_numpy(float)  # noqa: E712
    states = np.zeros((len(data), len(config.rates)))
    for _, group in data.groupby("session_id", sort=False):
        state = np.zeros(len(config.rates))
        previous_index = None
        for i, row in group.iterrows():
            if previous_index is not None and row.trial_index != previous_index + 1:
                state[:] = 0
            states[i] = state
            if row.action not in ("left", "right"):
                state[:] = 0
            elif abs(row.stimulus_contrast) > 0:
                inferred_side = (1 if row.action == "right" else -1) * (2 * float(row.correct) - 1)
                state += np.asarray(config.rates) * (inferred_side - state)
            previous_index = row.trial_index
    for j, rate in enumerate(config.rates):
        data[f"learned_side_{rate}"] = states[:, j]
    # prepare_trials provides validated canonical task cells and one-step nuisance terms.
    prepared = prepare_trials(data, AuditConfig(l2=config.l2))
    keys = pd.MultiIndex.from_frame(data.loc[keep, ["session_id", "trial_index"]])
    eligible = pd.MultiIndex.from_frame(prepared[["session_id", "trial_index"]]).isin(keys)
    result = prepared.loc[eligible].reset_index(drop=True)
    if result.empty:
        raise ValueError("No common contiguous-history trials")
    return result


def comparison_design(data: pd.DataFrame, model: str, config: ComparisonConfig) -> tuple[np.ndarray, list[str]]:
    """Nest the same two evidence interactions within stronger history controls."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model}")
    evidence = model.endswith("evidence") or model == "evidence_history"
    x, names = design_matrix(data, "evidence_history" if evidence else "outcome_history")
    extra = []
    if model.startswith(("long_", "combined_")):
        extra.extend(f"{kind}_lag_{lag}" for lag in range(2, config.lags + 1)
                     for kind in ("rewarded", "unrewarded"))
    if model.startswith(("learning_", "combined_")):
        extra.extend(f"learned_side_{rate}" for rate in config.rates)
    if extra:
        x = np.column_stack([x, data[extra].to_numpy(float)])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite predictors")
    return x, names + extra


def make_folds(data: pd.DataFrame, config: ComparisonConfig, unit: str = "subject") -> list[list[str]]:
    """Assign whole animals, or entire labs for sensitivity, to evaluation folds."""
    if unit not in ("subject", "lab") or data[unit].isna().any():
        raise ValueError("Missing grouping identities")
    units = np.array(sorted(data[unit].unique()))
    if unit == "lab":
        if len(units) < 2:
            raise ValueError("Need multiple labs")
        return [[str(value)] for value in units]
    if len(units) < config.folds:
        raise ValueError("Fewer subjects than folds")
    np.random.default_rng(config.seed).shuffle(units)
    return [part.tolist() for part in np.array_split(units, config.folds)]


def paired_gain(scores: pd.DataFrame, before: str, after: str) -> dict[str, object]:
    """Describe paired animal differences, conditional on these overlapping training fits."""
    table = scores.pivot(index="subject", columns="model", values="nll")
    delta = (table[before] - table[after]).dropna().to_numpy()
    rng = np.random.default_rng(4242)
    boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(1)
    return {"mean_gain": float(delta.mean()), "subject_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "subjects_improved": int((delta > 0).sum()), "subjects": len(delta)}


def compare(data: pd.DataFrame, config: ComparisonConfig, folds: list[list[str]],
            unit: str = "subject") -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit only complementary groups and retain per-trial losses plus every coefficient."""
    flat = [value for fold in folds for value in fold]
    if len(flat) != len(set(flat)) or set(flat) != set(data[unit]):
        raise ValueError("Folds must cover each grouping unit exactly once")
    records, fits = [], {}
    y = data.choice_right.to_numpy(float)
    for model in MODELS:
        x, names = comparison_design(data, model, config)
        loss = np.full(len(data), np.nan)
        probability = np.full(len(data), np.nan)
        fold_ids = np.full(len(data), -1)
        fits[model] = []
        for index, heldout in enumerate(folds):
            test = data[unit].isin(heldout).to_numpy()
            if not test.any() or test.all():
                raise ValueError("Empty fitting or evaluation partition")
            if set(data.loc[test, "subject"]) & set(data.loc[~test, "subject"]):
                raise ValueError("Animal crosses fitting and evaluation partitions")
            beta = fit_logistic(x[~test], y[~test], config.l2)
            z = x[test] @ beta
            loss[test] = np.logaddexp(0, z) - y[test] * z
            probability[test] = 1 / (1 + np.exp(-z))
            fold_ids[test] = index
            fits[model].append({"fold": index, "coefficients": dict(zip(names, beta.tolist())),
                                "training_trials": int((~test).sum()), "evaluation_trials": int(test.sum())})
        if not np.isfinite(loss).all():
            raise RuntimeError("Incomplete or invalid scores")
        frame = data[["subject", "lab", "session_id", "trial_index", "choice_right"]].copy()
        frame["model"], frame["nll"], frame["p_right"], frame["fold"] = model, loss, probability, fold_ids
        records.append(frame)
        print(f"scored {unit} folds: {model}", flush=True)
    predictions = pd.concat(records, ignore_index=True)
    subject = predictions.groupby(["subject", "lab", "model"], as_index=False).nll.mean()
    contrasts = {f"{after}_over_{before}": paired_gain(subject, before, after) for before, after in PAIRS}
    contrasts["combined_history_over_outcome_history"] = paired_gain(subject, "outcome_history", "combined_history")
    per_lab = subject.groupby(["lab", "model"], as_index=False).nll.mean()
    return predictions, {"contrasts": contrasts, "equal_animal_nll": subject.groupby("model").nll.mean().to_dict(),
                         "per_subject": subject.to_dict("records"), "per_lab": per_lab.to_dict("records"),
                         "fits": fits}
