"""Causal preference controls for an exploratory history-versus-bias diagnostic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from eval.competing_history import ComparisonConfig, comparison_design, prepare_comparison
from eval.history_audit import CONTRASTS, PRIORS, design_matrix, fit_logistic

__all__ = ["BiasConfig", "task_design", "bias_features", "evaluate_bias", "MODEL_NAMES"]
MODEL_NAMES = ("one_step", "long", "bias", "bias_long", "bias_evidence", "bias_long_evidence")


@dataclass(slots=True)
class BiasConfig:
    """Fixed prefix and filter controls; no held-out tuning."""

    prefix: int = 100
    min_prefix_choices: int = 80
    rates: tuple[float, ...] = (0.005, 0.02)
    l2: float = 1.0

    def __post_init__(self) -> None:
        if self.prefix < 5 or not 1 <= self.min_prefix_choices <= self.prefix:
            raise ValueError("Invalid prefix eligibility")
        if not np.isfinite(self.l2) or self.l2 <= 0:
            raise ValueError("Positive finite ridge required")
        if not self.rates or len(set(self.rates)) != len(self.rates) or any(not 0 < a < 1 for a in self.rates):
            raise ValueError("Distinct filter rates in (0,1) required")


def task_design(raw: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Categorical current stimulus/block controls without response information."""
    data = raw.copy()
    data["prior_right"] = data.block_prior.map(lambda b: b["p_right"])
    for name, levels in [("stimulus_contrast", CONTRASTS), ("prior_right", PRIORS)]:
        values = data[name].to_numpy(float)
        matches = np.isclose(values[:, None], np.array(levels)[None, :], atol=1e-8, rtol=0)
        if not (matches.sum(1) == 1).all():
            raise ValueError("Invalid task values")
        data[name] = np.array(levels)[matches.argmax(1)]
    return design_matrix(data, "stimulus_block")


def bias_features(raw: pd.DataFrame, task_beta: np.ndarray, config: BiasConfig) -> pd.DataFrame:
    """Prefix-only intercept and strictly past residual traces, resetting at gaps/omissions."""
    data = raw.sort_values(["session_id", "trial_index"]).reset_index(drop=True).copy()
    if data.duplicated(["session_id", "trial_index"]).any():
        raise ValueError("Duplicate trial keys")
    logits = task_design(data)[0] @ task_beta
    probabilities = expit(logits)
    valid = data.action.isin(["left", "right"]).to_numpy()
    choices = (data.action == "right").to_numpy(float)
    static = np.zeros(len(data))
    traces = np.zeros((len(data), len(config.rates)))
    eligible = np.zeros(len(data), dtype=bool)
    for _, group in data.groupby("session_id", sort=False):
        indices = group.index.to_numpy()
        prefix = indices[:config.prefix]
        prefix_valid = prefix[valid[prefix]]
        if len(indices) <= config.prefix or len(prefix_valid) < config.min_prefix_choices:
            continue
        z, y = logits[prefix_valid], choices[prefix_valid]
        fit = minimize_scalar(lambda b: float(np.sum(np.logaddexp(0, z + b) - y * (z + b)) + .5 * b*b),
                              bounds=(-4., 4.), method="bounded", options={"xatol": 1e-10})
        if not fit.success:
            raise RuntimeError("Prefix intercept fit failed")
        # Never expose the prefix-fitted value to an earlier prediction.
        static[indices[config.prefix:]] = fit.x
        eligible[indices[config.prefix:]] = True
        state = np.zeros(len(config.rates))
        previous = None
        for i in indices:
            trial = data.at[i, "trial_index"]
            if previous is not None and trial != previous + 1:
                state[:] = 0
            traces[i] = state
            if valid[i]:
                state = (1 - np.asarray(config.rates)) * state + np.asarray(config.rates) * (choices[i] - probabilities[i])
            else:
                state[:] = 0
            previous = trial
    data["prefix_bias"] = static
    data["after_prefix"] = eligible
    for j, rate in enumerate(config.rates):
        data[f"residual_trace_{rate}"] = traces[:, j]
    return data


def evaluate_bias(raw: pd.DataFrame, folds: list[list[str]], config: BiasConfig) -> tuple[pd.DataFrame, dict]:
    """Fit each nested model on other animals; derive all held-out states causally."""
    frame = raw.sort_values(["session_id", "trial_index"]).reset_index(drop=True)
    units = [s for fold in folds for s in fold]
    if len(units) != len(set(units)) or set(units) != set(frame.subject):
        raise ValueError("Folds must cover animals exactly once")
    cfg = ComparisonConfig(l2=config.l2)
    base_x, base_names = task_design(frame)
    valid = frame.action.isin(["left", "right"]).to_numpy()
    y_raw = (frame.action == "right").to_numpy(float)
    records, fits = [], []
    for fold, subjects in enumerate(folds):
        heldout = frame.subject.isin(subjects).to_numpy()
        training = ~heldout & valid
        base_beta = fit_logistic(base_x[training], y_raw[training], config.l2)
        data = prepare_comparison(bias_features(frame, base_beta, config), cfg)
        data = data[data.after_prefix].reset_index(drop=True)
        test = data.subject.isin(subjects).to_numpy()
        if not test.any() or test.all():
            raise ValueError("Empty evaluation or fitting sample")
        y = data.choice_right.to_numpy(float)
        fitted = {"fold": fold, "heldout_subjects": subjects,
                  "task_coefficients": dict(zip(base_names, base_beta.tolist())), "models": {}}
        for model in MODEL_NAMES:
            base = "long_history" if model in ("long", "bias_long", "bias_long_evidence") else "outcome_history"
            x, names = comparison_design(data, base, cfg)
            if model.startswith("bias"):
                additions = ["prefix_bias"] + [f"residual_trace_{rate}" for rate in config.rates]
                x = np.column_stack([x, data[additions].to_numpy(float)])
                names += additions
            if model.endswith("evidence"):
                weak = data.previous_weak.to_numpy(float)
                signed = data.previous_right_signed.to_numpy(float)
                success = data.previous_correct.to_numpy(float)
                x = np.column_stack([x, signed * weak * success, signed * weak * (1-success)])
                names += ["rewarded_choice_weak", "unrewarded_choice_weak"]
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite design")
            beta = fit_logistic(x[~test], y[~test], config.l2)
            z = x[test] @ beta
            loss = np.logaddexp(0, z) - y[test] * z
            out = data.loc[test, ["subject", "lab", "session_id", "trial_index", "choice_right"]].copy()
            out["model"], out["fold"], out["nll"], out["p_right"] = model, fold, loss, expit(z)
            records.append(out)
            fitted["models"][model] = {"coefficients": dict(zip(names, beta.tolist())),
                                       "training_trials": int((~test).sum()), "test_trials": int(test.sum())}
        fits.append(fitted)
        print(f"completed bias diagnostic fold {fold+1}/{len(folds)}", flush=True)
    predictions = pd.concat(records, ignore_index=True)
    scores = predictions.groupby(["subject", "lab", "model"], as_index=False).nll.mean()
    from eval.competing_history import paired_gain
    pairs = [("one_step", "long"), ("one_step", "bias"), ("bias", "bias_long"),
             ("bias", "bias_evidence"), ("bias_long", "bias_long_evidence")]
    summary = {"equal_animal_nll": scores.groupby("model").nll.mean().to_dict(),
               "per_subject": scores.to_dict("records"), "fits": fits,
               "comparisons": {f"{b}_over_{a}": paired_gain(scores, a, b) for a,b in pairs},
               "scored_trials": int(len(predictions)/len(MODEL_NAMES)),
               "subjects": scores.subject.nunique()}
    return predictions, summary
