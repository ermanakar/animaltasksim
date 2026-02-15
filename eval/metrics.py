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
from scipy.optimize import least_squares, minimize

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
    slope_unit: str = ""
    ceiling_fraction: float = 0.0  # fraction of difficulty levels pinned at max RT
    rt_range_ms: float = 0.0  # max(median RT) - min(median RT) across levels
    corrected_slope: float | None = None  # slope with ceiling levels removed


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

    filtered["choice_right"] = (filtered["action"] == "right").astype(float)

    # Exclude zero-coherence trials from slope fitting
    f_fit = filtered[filtered[f"stimulus_{stimulus_key}"] != 0]
    if f_fit.empty:
        return PsychometricMetrics(np.nan, np.nan, np.nan, np.nan)

    df_for_fit = f_fit.copy()
    df_for_fit["stimulus"] = df_for_fit[f"stimulus_{stimulus_key}"]
    df_for_fit["choice_right"] = (df_for_fit["action"] == "right").astype(float)
    grouped = df_for_fit.groupby("stimulus")["choice_right"].agg(["mean", "count"]).reset_index()

    xdata = grouped["stimulus"].values.astype(float)
    ydata = grouped["mean"].values.astype(float)
    weights = grouped["count"].values.astype(float)

    if len(xdata) < 2 or np.allclose(xdata, xdata[0]):
        return PsychometricMetrics(np.nan, np.nan, np.nan, np.nan)

    initial = np.array([0.0, 5.0, 0.02, 0.02], dtype=float)
    lower_bounds = np.array([-np.inf, 0.01, 0.0, 0.0], dtype=float)
    upper_bounds = np.array([np.inf, 50.0, 0.49, 0.49], dtype=float)
    weight_scale = np.sqrt(np.clip(weights, 1.0, None))

    def _residuals(params: np.ndarray) -> np.ndarray:
        preds = _logistic_function(xdata, params[0], params[1], params[2], params[3])
        return (preds - ydata) * weight_scale

    try:
        result = least_squares(
            _residuals,
            x0=initial,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=10000,
        )
    except Exception:  # pragma: no cover - rare fit failure fallback
        result = None

    if result is None or not result.success:
        bias = float(np.interp(0.5, ydata, xdata)) if np.isfinite(ydata).all() else np.nan
        slope = np.nan
        lapse_low = float(max(0.0, 1.0 - ydata.max()))
        lapse_high = float(max(0.0, ydata.min()))
    else:
        bias, slope, lapse_low, lapse_high = result.x

    # Lapse from zero-coh only:
    zero = filtered[filtered[f"stimulus_{stimulus_key}"] == 0]
    lapse = float(zero["choice_right"].mean()) if not zero.empty else np.nan
    if not np.isnan(lapse):
        lapse_low = lapse
        lapse_high = 1 - lapse

    return PsychometricMetrics(float(slope), float(bias), float(lapse_low), float(lapse_high))


def compute_chronometric(df: pd.DataFrame, stimulus_key: str = "coherence") -> ChronometricMetrics:
    data = df.copy()
    data["rt_used"] = data["rt_ms"].fillna(0.0)
    mask = data["rt_used"] > 0.0
    data = data[mask]
    if data.empty:
        return ChronometricMetrics(np.nan, np.nan, {}, "")

    data["difficulty"] = np.abs(data[f"stimulus_{stimulus_key}"])
    data["rt_used"] = data["rt_used"]
    grouped = data.groupby("difficulty")["rt_ms"].median().sort_index()
    if len(grouped) < 2:
        intercept = float(grouped.iloc[0]) if not grouped.empty else np.nan
        rt_dict = {float(k): float(v) for k, v in grouped.items()}
        return ChronometricMetrics(intercept, np.nan, rt_dict, f"ms_per_10pct_{stimulus_key}")

    raw_x = grouped.index.values.astype(float)
    y = grouped.values.astype(float)

    # Detect RT ceiling saturation: levels where median RT equals the maximum
    max_rt = float(np.max(y))
    min_rt = float(np.min(y))
    rt_range = max_rt - min_rt
    # A level is "at ceiling" if its median RT is within 1% of the max
    ceiling_tol = max(1.0, max_rt * 0.01)
    n_at_ceiling = int(np.sum(np.abs(y - max_rt) < ceiling_tol))
    ceiling_fraction = n_at_ceiling / len(y) if len(y) > 0 else 0.0

    percent_mode = False
    if np.nanmax(np.abs(raw_x)) <= 1.0 + 1e-6:
        x = raw_x * 100.0
        percent_mode = True
    elif np.nanmax(np.abs(raw_x)) <= 100.0 + 1e-6:
        x = raw_x
        percent_mode = True
    else:
        x = raw_x

    slope_raw, intercept = np.polyfit(x, y, deg=1)
    # Convert grouped dict keys/values explicitly to avoid type issues
    rt_dict: dict[float, float] = {float(k): float(v) for k, v in grouped.items()}  # type: ignore[arg-type]
    if percent_mode:
        slope = float(slope_raw * 10.0)
        slope_unit = f"ms_per_10pct_{stimulus_key}"
    else:
        slope = float(slope_raw / 10.0)
        slope_unit = f"ms_per_10_{stimulus_key}_units"

    # Ceiling-corrected slope: exclude levels at ceiling and refit
    # Only compute when ceiling fraction is concerning (≥ 2 levels at ceiling)
    corrected_slope: float | None = None
    if n_at_ceiling >= 2:
        not_at_ceiling = np.abs(y - max_rt) >= ceiling_tol
        x_clean = x[not_at_ceiling]
        y_clean = y[not_at_ceiling]
        if len(x_clean) >= 2:
            corr_raw, _ = np.polyfit(x_clean, y_clean, deg=1)
            if percent_mode:
                corrected_slope = float(corr_raw * 10.0)
            else:
                corrected_slope = float(corr_raw / 10.0)

    return ChronometricMetrics(
        float(intercept), slope, rt_dict, slope_unit,
        ceiling_fraction, rt_range, corrected_slope,
    )


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
    if isinstance(obj, bool):
        return obj
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


def _is_finite(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, numbers.Real):
        return math.isfinite(float(value))
    return False


def _quality_flags(metrics: dict[str, object], task: str, is_choice_only: bool = False) -> dict[str, bool]:
    psychometric = metrics.get("psychometric", {}) or {}
    history = metrics.get("history", {}) or {}
    chronometric = metrics.get("chronometric", {}) or {}

    bias = psychometric.get("bias")
    win_stay = history.get("win_stay")
    lose_shift = history.get("lose_shift")
    sticky = history.get("sticky_choice")
    slope = chronometric.get("slope_ms_per_unit")
    intercept = chronometric.get("intercept_ms")

    bias_ok = _is_finite(bias) and abs(float(bias)) <= 30.0
    history_ok = (
        _is_finite(win_stay)
        and _is_finite(lose_shift)
        and _is_finite(sticky)
        and float(win_stay) < 0.95
        and float(sticky) < 0.95
        and float(lose_shift) > 0.05
    )
    rt_ok = _is_finite(intercept) and 150.0 <= float(intercept) <= 3500.0

    ceiling_frac = chronometric.get("ceiling_fraction", 0.0)
    ceiling_warning = _is_finite(ceiling_frac) and float(ceiling_frac) >= 0.5

    # When ceiling is present, prefer the corrected slope for quality assessment
    corrected = chronometric.get("corrected_slope")
    effective_slope = corrected if (_is_finite(corrected) and ceiling_warning) else slope

    chrono_ok = metrics.get("chronometric", {}).get("slope_ms_per_unit", 0.0) is not None
    if is_choice_only:
        chrono_ok = True
    elif task == "ibl_2afc":
        chrono_ok = _is_finite(effective_slope) and float(effective_slope) <= -10.0
    elif task == "rdm":
        chrono_ok = _is_finite(effective_slope) and float(effective_slope) <= -5.0
    else:
        chrono_ok = _is_finite(effective_slope)

    degenerate = False
    if not bias_ok:
        degenerate = True
    if not history_ok:
        degenerate = True
    if not rt_ok:
        degenerate = True
    if not chrono_ok:
        degenerate = True
    return {
        "bias_ok": bias_ok,
        "history_ok": history_ok,
        "rt_ok": rt_ok,
        "chronometric_ok": chrono_ok,
        "rt_ceiling_warning": ceiling_warning,
        "degenerate": degenerate,
    }


def compute_all_metrics(df: pd.DataFrame, task: str, is_choice_only: bool = False) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if not df.empty:
        metrics["p_right_overall"] = (df["action"] == "right").mean()
        committed = df[df["action"].isin(["left", "right"])]
        if len(committed) > 0:
            metrics["p_right_committed"] = (committed["action"] == "right").mean()
        else:
            metrics["p_right_committed"] = float("nan")
        metrics["commit_rate"] = len(committed) / len(df)
        metrics["rt_variance"] = df["rt_ms"].var()

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
    metrics["quality"] = _quality_flags(metrics, task, is_choice_only)
    return _sanitize(metrics)  # type: ignore[return-value]


def load_and_compute(path: str | Path, is_choice_only: bool = False) -> dict[str, object]:
    df = load_trials(path)
    if df.empty:
        return {}
    task = df["task"].iloc[0]
    return compute_all_metrics(df, task, is_choice_only)


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
