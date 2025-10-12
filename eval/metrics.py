"""Evaluation metrics for AnimalTaskSim trial logs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import math
import numbers
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize

from eval.schema_validator import TrialRecord


@dataclass(slots=True)
class PsychometricMetrics:
    slope: float
    bias: float
    lapse_low: float
    lapse_high: float


@dataclass(slots=True)
class ChronometricMetrics:
    intercept_ms: float
    slope_ms_per_unit: float
    rt_by_level: dict[float, float]


@dataclass(slots=True)
class HistoryMetrics:
    win_stay: float
    lose_shift: float
    sticky_choice: float
    prev_choice_beta: float
    prev_correct_beta: float


def load_trials(path: str | Path) -> pd.DataFrame:
    """Load a validated `.ndjson` log into a DataFrame."""

    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            trial = TrialRecord.model_validate_json(raw)
            record = trial.model_dump()
            task = record.get("task", "").lower()
            if task in {"ibl2afc", "ibl-2afc", "ibl_2afc"}:
                record["task"] = "ibl_2afc"
            elif task in {"rdm", "rdm_task", "rdm_macaque"}:
                record["task"] = "rdm"
            else:
                record["task"] = task
            stim = record.pop("stimulus")
            for key, value in stim.items():
                record[f"stimulus_{key}"] = value
            prev = record.pop("prev")
            if prev is not None:
                for key, value in prev.items():
                    record[f"prev_{key}"] = value
            else:
                record["prev_action"] = None
                record["prev_reward"] = None
                record["prev_correct"] = None
            record.setdefault("prev_reward", None)
            record.setdefault("prev_correct", None)
            records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    df.sort_values(["session_id", "trial_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Normalize actions: handle both string ("left"/"right") and numeric (0=right, 1=left) formats
    if "action" in df.columns:
        # Convert numeric actions to strings if present
        df["action"] = df["action"].apply(lambda x: "left" if x == 1 else ("right" if x == 0 else x))
        # Also normalize prev_action if present
        if "prev_action" in df.columns:
            df["prev_action"] = df["prev_action"].apply(
                lambda x: "left" if x == 1 else ("right" if x == 0 else x) if pd.notna(x) else x
            )
    
    return df


def _logistic_function(x: np.ndarray, bias: float, slope: float, lapse_low: float, lapse_high: float) -> np.ndarray:
    lapse_low = np.clip(lapse_low, 0.0, 0.49)
    lapse_high = np.clip(lapse_high, 0.0, 0.49)
    core = 1.0 / (1.0 + np.exp(-(x - bias) * slope))
    return lapse_low + (1.0 - lapse_low - lapse_high) * core


def compute_psychometric(df: pd.DataFrame, stimulus_key: str = "contrast") -> PsychometricMetrics:
    filtered = df[df[f"stimulus_{stimulus_key}"].notnull()].copy()
    if filtered.empty:
        return PsychometricMetrics(np.nan, np.nan, np.nan, np.nan)

    filtered["stimulus"] = filtered[f"stimulus_{stimulus_key}"]
    filtered["choice_right"] = (filtered["action"] == "right").astype(float)
    grouped = filtered.groupby("stimulus")["choice_right"].agg(["mean", "count"]).reset_index()

    xdata = grouped["stimulus"].values.astype(float)
    ydata = grouped["mean"].values.astype(float)
    weights = grouped["count"].values.astype(float)

    if len(xdata) < 2 or np.allclose(xdata, xdata[0]):
        return PsychometricMetrics(np.nan, np.nan, np.nan, np.nan)

    try:
        popt, _ = curve_fit(
            _logistic_function,
            xdata,
            ydata,
            sigma=1.0 / np.clip(weights, 1, None),
            p0=[0.0, 5.0, 0.02, 0.02],
            bounds=([-np.inf, 0.01, 0.0, 0.0], [np.inf, 50.0, 0.49, 0.49]),
            maxfev=1000,
        )
        bias, slope, lapse_low, lapse_high = popt
    except Exception:  # pragma: no cover - rare fit failure fallback
        bias = float(np.interp(0.5, ydata, xdata)) if np.isfinite(ydata).all() else np.nan
        slope = np.nan
        lapse_low = float(max(0.0, 1.0 - ydata.max()))
        lapse_high = float(max(0.0, ydata.min()))

    return PsychometricMetrics(float(slope), float(bias), float(lapse_low), float(lapse_high))


def compute_chronometric(df: pd.DataFrame, stimulus_key: str = "coherence") -> ChronometricMetrics:
    data = df.copy()
    data["rt_used"] = data["rt_ms"].fillna(0.0)
    mask = data["rt_used"] > 0.0
    data = data[mask]
    if data.empty:
        return ChronometricMetrics(np.nan, np.nan, {})

    data["difficulty"] = np.abs(data[f"stimulus_{stimulus_key}"])
    data["rt_used"] = data["rt_used"]
    grouped = data.groupby("difficulty")["rt_ms"].median().sort_index()
    if len(grouped) < 2:
        intercept = float(grouped.iloc[0]) if not grouped.empty else np.nan
        return ChronometricMetrics(intercept, np.nan, grouped.to_dict())

    x = grouped.index.values.astype(float)
    y = grouped.values.astype(float)
    slope, intercept = np.polyfit(x, y, deg=1)
    # Convert grouped dict keys/values explicitly to avoid type issues
    rt_dict: dict[float, float] = {float(k): float(v) for k, v in grouped.items()}  # type: ignore[arg-type]
    return ChronometricMetrics(float(intercept), float(slope), rt_dict)


def compute_history_metrics(df: pd.DataFrame) -> HistoryMetrics:
    data = df.copy()
    data["right_choice"] = (data["action"] == "right").astype(int)
    prev_actions = data["prev_action"]
    mask_prev = prev_actions.notnull()
    same = np.full(len(data), np.nan, dtype=float)
    same[mask_prev.to_numpy()] = (
        data.loc[mask_prev, "action"] == prev_actions[mask_prev]
    ).astype(float)
    data["same_as_prev"] = same

    valid_prev = data[data["prev_action"].notnull()].copy()
    if "prev_correct" not in valid_prev or valid_prev["prev_correct"].isnull().all():
        return HistoryMetrics(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    # Fix FutureWarning by explicitly converting to float before fillna
    valid_prev["prev_correct"] = valid_prev["prev_correct"].astype(float).fillna(0.0)
    win_trials = valid_prev[valid_prev["prev_correct"] > 0.5]
    lose_trials = valid_prev[valid_prev["prev_correct"] <= 0.5]

    def _ratio(values: pd.Series, transform: Callable[[pd.Series], pd.Series] | None = None) -> float:
        if values.empty:
            return float("nan")
        data = values
        if transform is not None:
            data = transform(data)
        return float(np.nanmean(data))

    win_stay = _ratio(win_trials["same_as_prev"])
    lose_shift = _ratio(lose_trials["same_as_prev"], lambda x: 1.0 - x)
    sticky = _ratio(valid_prev["same_as_prev"])

    logistic_subset = valid_prev[valid_prev["action"].isin(["left", "right"])].copy()
    logistic_subset = logistic_subset[logistic_subset["prev_action"].isin(["left", "right"])].copy()

    if logistic_subset.empty:
        return HistoryMetrics(win_stay, lose_shift, sticky, float("nan"), float("nan"))

    prev_choice = np.where(logistic_subset["prev_action"] == "right", 1.0, -1.0)
    prev_correct = logistic_subset["prev_correct"].fillna(0.0).astype(float).to_numpy()
    X = np.column_stack([np.ones(len(logistic_subset)), prev_choice, prev_correct])
    y = logistic_subset["right_choice"].values.astype(float)

    def _neg_loglik(theta: np.ndarray) -> float:
        z = X @ theta
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-9
        return -float(np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))

    res = minimize(_neg_loglik, x0=np.zeros(X.shape[1]), method="BFGS")
    if not res.success:
        prev_choice_beta = float("nan")
        prev_correct_beta = float("nan")
    else:
        _, prev_choice_beta, prev_correct_beta = res.x

    return HistoryMetrics(win_stay, lose_shift, sticky, float(prev_choice_beta), float(prev_correct_beta))


def _sanitize(obj: object) -> object:
    if isinstance(obj, numbers.Real):
        value = float(obj)
        if not math.isfinite(value):
            return None
        return value
    if isinstance(obj, dict):
        return {key: _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize(value) for value in obj.tolist()]
    return obj


def compute_all_metrics(df: pd.DataFrame, task: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if task == "ibl_2afc":
        metrics["psychometric"] = _sanitize(asdict(compute_psychometric(df, stimulus_key="contrast")))
        metrics["chronometric"] = _sanitize(asdict(compute_chronometric(df, stimulus_key="contrast")))
        metrics["history"] = _sanitize(asdict(compute_history_metrics(df)))
    elif task == "rdm":
        metrics["psychometric"] = _sanitize(asdict(compute_psychometric(df, stimulus_key="coherence")))
        metrics["chronometric"] = _sanitize(asdict(compute_chronometric(df, stimulus_key="coherence")))
        metrics["history"] = _sanitize(asdict(compute_history_metrics(df)))
    else:  # pragma: no cover - defensive fallback for future tasks
        metrics["history"] = _sanitize(asdict(compute_history_metrics(df)))
    return metrics


def load_and_compute(path: str | Path) -> dict[str, object]:
    df = load_trials(path)
    if df.empty:
        return {}
    task = df["task"].iloc[0]
    return compute_all_metrics(df, task)


__all__ = [
    "ChronometricMetrics",
    "HistoryMetrics",
    "PsychometricMetrics",
    "compute_all_metrics",
    "compute_chronometric",
    "compute_history_metrics",
    "compute_psychometric",
    "load_and_compute",
    "load_trials",
]
