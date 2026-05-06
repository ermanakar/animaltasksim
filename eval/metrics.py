"""Evaluation metrics for AnimalTaskSim trial logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numbers
from pathlib import Path
from typing import Callable

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


@dataclass(slots=True)
class AdaptiveControlProbeMetrics:
    retry_after_failure_weak: float
    retry_after_failure_strong: float
    switch_after_failure_weak: float
    switch_after_failure_strong: float
    weak_failure_count: int
    strong_failure_count: int


@dataclass(slots=True)
class ExplorationProbeMetrics:
    switch_after_streak_weak: float
    switch_after_streak_strong: float
    weak_streak_count: int
    strong_streak_count: int
    switch_after_fresh_weak: float = float("nan")
    switch_after_fresh_overall: float = float("nan")
    switch_after_stale_overall: float = float("nan")
    stale_switch_lift_weak: float = float("nan")
    stale_switch_lift_overall: float = float("nan")
    fresh_weak_count: int = 0
    fresh_count: int = 0
    stale_count: int = 0
    switch_after_unrewarded_streak_weak: float = float("nan")
    switch_after_unrewarded_fresh_weak: float = float("nan")
    unrewarded_switch_lift_weak: float = float("nan")
    unrewarded_streak_weak_count: int = 0
    unrewarded_fresh_weak_count: int = 0
    switch_after_volatile_weak: float = float("nan")
    switch_after_stable_weak: float = float("nan")
    volatile_switch_lift_weak: float = float("nan")
    volatile_weak_count: int = 0
    stable_weak_count: int = 0


@dataclass(slots=True)
class BlockSwitchProbeMetrics:
    switch_count: int
    post_switch_trial_count: int
    early_trial_count: int
    late_trial_count: int
    early_new_prior_choice_rate: float
    late_new_prior_choice_rate: float
    adaptation_lift: float
    early_perseverative_choice_rate: float
    low_contrast_trial_count: int
    low_contrast_new_prior_choice_rate: float
    zero_contrast_trial_count: int
    zero_contrast_new_prior_choice_rate: float


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

    # Use all stimulus levels (including zero) for the full sigmoid fit
    df_for_fit = filtered.copy()
    df_for_fit["stimulus"] = df_for_fit[f"stimulus_{stimulus_key}"]
    grouped = df_for_fit.groupby("stimulus")["choice_right"].agg(["mean", "count"]).reset_index()

    xdata = grouped["stimulus"].values.astype(float)
    ydata = grouped["mean"].values.astype(float)
    weights = grouped["count"].values.astype(float)

    if len(xdata) < 2 or np.allclose(xdata, xdata[0]):
        return PsychometricMetrics(np.nan, np.nan, np.nan, np.nan)

    # Estimate initial slope from data range for better convergence
    y_range = ydata.max() - ydata.min()
    x_range = xdata.max() - xdata.min()
    slope_init = max(2.0, 4.0 * y_range / max(x_range, 1e-6))
    lapse_init = max(0.001, min(0.15, ydata.min()))

    initial = np.array([0.0, slope_init, lapse_init, lapse_init], dtype=float)
    lower_bounds = np.array([-np.inf, 0.01, 0.0, 0.0], dtype=float)
    upper_bounds = np.array([np.inf, 200.0, 0.5, 0.5], dtype=float)
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
        lapse_low = float(max(0.0, ydata.min()))
        lapse_high = float(max(0.0, 1.0 - ydata.max()))
    else:
        bias, slope, lapse_low, lapse_high = result.x

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


def compute_adaptive_control_probe_metrics(df: pd.DataFrame) -> AdaptiveControlProbeMetrics:
    """Measure retry versus switch after failure split by evidence strength."""
    if df.empty:
        return AdaptiveControlProbeMetrics(np.nan, np.nan, np.nan, np.nan, 0, 0)

    task = str(df["task"].iloc[0]) if "task" in df.columns and not df.empty else ""
    if task == "ibl_2afc":
        stimulus_key = "stimulus_contrast"
    elif task == "rdm":
        stimulus_key = "stimulus_coherence"
    elif "stimulus_contrast" in df.columns:
        stimulus_key = "stimulus_contrast"
    elif "stimulus_coherence" in df.columns:
        stimulus_key = "stimulus_coherence"
    else:
        return AdaptiveControlProbeMetrics(np.nan, np.nan, np.nan, np.nan, 0, 0)

    data = df.copy()
    data = data[data["action"].isin(["left", "right"])].copy()
    data = data[data["prev_action"].isin(["left", "right"])].copy()
    if data.empty:
        return AdaptiveControlProbeMetrics(np.nan, np.nan, np.nan, np.nan, 0, 0)

    prev_correct = pd.to_numeric(data["prev_correct"], errors="coerce")
    data = data[prev_correct.notnull()].copy()
    if data.empty:
        return AdaptiveControlProbeMetrics(np.nan, np.nan, np.nan, np.nan, 0, 0)
    data["prev_correct"] = prev_correct.loc[data.index].astype(float)
    data["same_as_prev"] = (data["action"] == data["prev_action"]).astype(float)
    data["difficulty_abs"] = pd.to_numeric(data[stimulus_key], errors="coerce").abs()
    data = data[data["difficulty_abs"].notnull()].copy()
    failures = data[data["prev_correct"] <= 0.5].copy()
    if failures.empty:
        return AdaptiveControlProbeMetrics(np.nan, np.nan, np.nan, np.nan, 0, 0)

    difficulty_levels = np.sort(failures["difficulty_abs"].unique().astype(float))
    split_threshold = float(np.median(difficulty_levels))
    weak_failures = failures[failures["difficulty_abs"] <= split_threshold].copy()
    strong_failures = failures[failures["difficulty_abs"] > split_threshold].copy()

    def _ratio(values: pd.Series, transform: Callable[[pd.Series], pd.Series] | None = None) -> float:
        if values.empty:
            return float("nan")
        data_series = transform(values) if transform is not None else values
        return float(np.nanmean(data_series))

    return AdaptiveControlProbeMetrics(
        retry_after_failure_weak=_ratio(weak_failures["same_as_prev"]),
        retry_after_failure_strong=_ratio(strong_failures["same_as_prev"]),
        switch_after_failure_weak=_ratio(weak_failures["same_as_prev"], lambda x: 1.0 - x),
        switch_after_failure_strong=_ratio(strong_failures["same_as_prev"], lambda x: 1.0 - x),
        weak_failure_count=int(len(weak_failures)),
        strong_failure_count=int(len(strong_failures)),
    )


def compute_exploration_probe_metrics(
    df: pd.DataFrame,
    streak_length: int = 3,
    unrewarded_streak_length: int = 2,
    volatility_window: int = 4,
) -> ExplorationProbeMetrics:
    """Measure structured switching after stale, unrewarded, or volatile history."""
    if df.empty:
        return ExplorationProbeMetrics(np.nan, np.nan, 0, 0)

    task = str(df["task"].iloc[0]) if "task" in df.columns and not df.empty else ""
    if task == "ibl_2afc":
        stimulus_key = "stimulus_contrast"
    elif task == "rdm":
        stimulus_key = "stimulus_coherence"
    elif "stimulus_contrast" in df.columns:
        stimulus_key = "stimulus_contrast"
    elif "stimulus_coherence" in df.columns:
        stimulus_key = "stimulus_coherence"
    else:
        return ExplorationProbeMetrics(np.nan, np.nan, 0, 0)

    data = df.copy()
    data = data[data["action"].isin(["left", "right"])].copy()
    if data.empty:
        return ExplorationProbeMetrics(np.nan, np.nan, 0, 0)
    data["difficulty_abs"] = pd.to_numeric(data[stimulus_key], errors="coerce").abs()
    data = data[data["difficulty_abs"].notnull()].copy()
    if data.empty:
        return ExplorationProbeMetrics(np.nan, np.nan, 0, 0)

    difficulty_levels = np.sort(data["difficulty_abs"].unique().astype(float))
    split_threshold = float(np.median(difficulty_levels))
    data["is_weak_evidence"] = data["difficulty_abs"] <= split_threshold
    sort_columns = [column for column in ("session_id", "trial_index") if column in data.columns]
    if sort_columns:
        data = data.sort_values(sort_columns).copy()

    run_lengths: list[int] = []
    failure_run_lengths: list[int] = []
    volatility_flags: list[bool] = []
    groupby_key = "session_id" if "session_id" in data.columns else None
    grouped_rows = data.groupby(groupby_key, sort=False) if groupby_key is not None else [(None, data)]
    for _, session_rows in grouped_rows:
        current_rewarded_streak = 0
        current_rewarded_action: str | None = None
        current_failure_streak = 0
        current_failure_action: str | None = None
        previous_outcomes: list[float] = []
        for _, row in session_rows.iterrows():
            action = row.get("action")
            run_lengths.append(current_rewarded_streak)
            failure_run_lengths.append(current_failure_streak)
            recent = previous_outcomes[-volatility_window:]
            volatility_flags.append(
                len(recent) >= volatility_window
                and any(outcome > 0.5 for outcome in recent)
                and any(outcome <= 0.5 for outcome in recent)
            )

            success = _trial_success(row)
            if action in {"left", "right"} and success:
                if current_rewarded_action == action and current_rewarded_streak > 0:
                    current_rewarded_streak += 1
                else:
                    current_rewarded_streak = 1
                current_rewarded_action = action
            else:
                current_rewarded_streak = 0
                current_rewarded_action = None

            is_weak_failure = (
                action in {"left", "right"}
                and not success
                and bool(row.get("is_weak_evidence", False))
            )
            if is_weak_failure:
                if current_failure_action == action and current_failure_streak > 0:
                    current_failure_streak += 1
                else:
                    current_failure_streak = 1
                current_failure_action = action
            else:
                current_failure_streak = 0
                current_failure_action = None
            previous_outcomes.append(1.0 if success else 0.0)

    data["previous_correct_streak"] = run_lengths
    data["previous_weak_failure_streak"] = failure_run_lengths
    data["previous_outcome_volatility"] = volatility_flags
    data = data[data["prev_action"].isin(["left", "right"])].copy()
    if data.empty:
        return ExplorationProbeMetrics(np.nan, np.nan, 0, 0)
    data["same_as_prev"] = (data["action"] == data["prev_action"]).astype(float)
    data["is_stale_streak"] = data["previous_correct_streak"] >= streak_length
    data["is_unrewarded_streak"] = data["previous_weak_failure_streak"] >= unrewarded_streak_length

    stale = data[data["is_stale_streak"]].copy()
    fresh = data[~data["is_stale_streak"]].copy()
    weak_streak = stale[stale["is_weak_evidence"]].copy()
    strong_streak = stale[~stale["is_weak_evidence"]].copy()
    fresh_weak = fresh[fresh["is_weak_evidence"]].copy()
    weak_unrewarded_streak = data[data["is_weak_evidence"] & data["is_unrewarded_streak"]].copy()
    weak_unrewarded_fresh = data[data["is_weak_evidence"] & ~data["is_unrewarded_streak"]].copy()
    weak_volatile = data[data["is_weak_evidence"] & data["previous_outcome_volatility"]].copy()
    weak_stable = data[data["is_weak_evidence"] & ~data["previous_outcome_volatility"]].copy()

    def _ratio(values: pd.Series, transform: Callable[[pd.Series], pd.Series] | None = None) -> float:
        if values.empty:
            return float("nan")
        data_series = transform(values) if transform is not None else values
        return float(np.nanmean(data_series))

    def _difference(high: float, low: float) -> float:
        if not math.isfinite(high) or not math.isfinite(low):
            return float("nan")
        return high - low

    switch_after_streak_weak = _ratio(weak_streak["same_as_prev"], lambda x: 1.0 - x)
    switch_after_streak_strong = _ratio(strong_streak["same_as_prev"], lambda x: 1.0 - x)
    switch_after_fresh_weak = _ratio(fresh_weak["same_as_prev"], lambda x: 1.0 - x)
    switch_after_fresh_overall = _ratio(fresh["same_as_prev"], lambda x: 1.0 - x)
    switch_after_stale_overall = _ratio(stale["same_as_prev"], lambda x: 1.0 - x)
    switch_after_unrewarded_streak_weak = _ratio(
        weak_unrewarded_streak["same_as_prev"],
        lambda x: 1.0 - x,
    )
    switch_after_unrewarded_fresh_weak = _ratio(
        weak_unrewarded_fresh["same_as_prev"],
        lambda x: 1.0 - x,
    )
    switch_after_volatile_weak = _ratio(weak_volatile["same_as_prev"], lambda x: 1.0 - x)
    switch_after_stable_weak = _ratio(weak_stable["same_as_prev"], lambda x: 1.0 - x)

    return ExplorationProbeMetrics(
        switch_after_streak_weak=switch_after_streak_weak,
        switch_after_streak_strong=switch_after_streak_strong,
        weak_streak_count=int(len(weak_streak)),
        strong_streak_count=int(len(strong_streak)),
        switch_after_fresh_weak=switch_after_fresh_weak,
        switch_after_fresh_overall=switch_after_fresh_overall,
        switch_after_stale_overall=switch_after_stale_overall,
        stale_switch_lift_weak=_difference(switch_after_streak_weak, switch_after_fresh_weak),
        stale_switch_lift_overall=_difference(switch_after_stale_overall, switch_after_fresh_overall),
        fresh_weak_count=int(len(fresh_weak)),
        fresh_count=int(len(fresh)),
        stale_count=int(len(stale)),
        switch_after_unrewarded_streak_weak=switch_after_unrewarded_streak_weak,
        switch_after_unrewarded_fresh_weak=switch_after_unrewarded_fresh_weak,
        unrewarded_switch_lift_weak=_difference(
            switch_after_unrewarded_streak_weak,
            switch_after_unrewarded_fresh_weak,
        ),
        unrewarded_streak_weak_count=int(len(weak_unrewarded_streak)),
        unrewarded_fresh_weak_count=int(len(weak_unrewarded_fresh)),
        switch_after_volatile_weak=switch_after_volatile_weak,
        switch_after_stable_weak=switch_after_stable_weak,
        volatile_switch_lift_weak=_difference(switch_after_volatile_weak, switch_after_stable_weak),
        volatile_weak_count=int(len(weak_volatile)),
        stable_weak_count=int(len(weak_stable)),
    )


def compute_block_switch_probe_metrics(
    df: pd.DataFrame,
    post_switch_window: int = 10,
    early_window: int = 5,
    low_contrast_threshold: float = 0.125,
) -> BlockSwitchProbeMetrics:
    """Measure adaptation after hidden IBL block-prior reversals."""
    if df.empty or "block_prior" not in df.columns:
        return _empty_block_switch_metrics()

    data = df.copy()
    data = data[data["action"].isin(["left", "right"])].copy()
    if data.empty:
        return _empty_block_switch_metrics()

    data["block_p_right"] = data["block_prior"].apply(_block_prior_p_right)
    data = data[data["block_p_right"].notnull()].copy()
    if data.empty:
        return _empty_block_switch_metrics()

    stimulus_key = "stimulus_contrast" if "stimulus_contrast" in data.columns else None
    if stimulus_key is not None:
        data["difficulty_abs"] = pd.to_numeric(data[stimulus_key], errors="coerce").abs()
    else:
        data["difficulty_abs"] = np.nan

    sort_columns = [column for column in ("session_id", "trial_index") if column in data.columns]
    if sort_columns:
        data = data.sort_values(sort_columns).copy()

    event_rows: list[dict[str, object]] = []
    groupby_key = "session_id" if "session_id" in data.columns else None
    grouped_rows = data.groupby(groupby_key, sort=False) if groupby_key is not None else [(None, data)]
    switch_count = 0
    for _, session_rows in grouped_rows:
        session_rows = session_rows.reset_index(drop=True)
        priors = session_rows["block_p_right"].to_numpy(dtype=float)
        previous_prior = priors[0] if len(priors) else np.nan
        for position in range(1, len(session_rows)):
            new_prior = priors[position]
            if not math.isfinite(previous_prior) or not math.isfinite(new_prior):
                previous_prior = new_prior
                continue
            old_direction = _prior_direction(previous_prior)
            new_direction = _prior_direction(new_prior)
            if old_direction is None or new_direction is None or old_direction == new_direction:
                previous_prior = new_prior
                continue

            switch_count += 1
            window = session_rows.iloc[position : position + post_switch_window].copy()
            window["offset_after_switch"] = np.arange(1, len(window) + 1)
            for _, row in window.iterrows():
                row_prior_direction = _prior_direction(float(row["block_p_right"]))
                if row_prior_direction != new_direction:
                    break
                action = row["action"]
                if action not in {"left", "right"}:
                    continue
                event_rows.append(
                    {
                        "offset_after_switch": int(row["offset_after_switch"]),
                        "new_prior_choice": action == new_direction,
                        "old_prior_choice": action == old_direction,
                        "difficulty_abs": row.get("difficulty_abs"),
                    }
                )
            previous_prior = new_prior

    if not event_rows:
        return _empty_block_switch_metrics(switch_count=switch_count)

    events = pd.DataFrame.from_records(event_rows)
    events["new_prior_choice"] = events["new_prior_choice"].astype(float)
    events["old_prior_choice"] = events["old_prior_choice"].astype(float)
    early = events[events["offset_after_switch"] <= early_window]
    late = events[events["offset_after_switch"] > early_window]
    low_contrast = events[events["difficulty_abs"].notnull() & (events["difficulty_abs"] <= low_contrast_threshold)]
    zero_contrast = events[events["difficulty_abs"].notnull() & np.isclose(events["difficulty_abs"], 0.0)]

    early_rate = _mean_or_nan(early["new_prior_choice"])
    late_rate = _mean_or_nan(late["new_prior_choice"])
    adaptation_lift = late_rate - early_rate if math.isfinite(early_rate) and math.isfinite(late_rate) else np.nan

    return BlockSwitchProbeMetrics(
        switch_count=switch_count,
        post_switch_trial_count=int(len(events)),
        early_trial_count=int(len(early)),
        late_trial_count=int(len(late)),
        early_new_prior_choice_rate=early_rate,
        late_new_prior_choice_rate=late_rate,
        adaptation_lift=float(adaptation_lift),
        early_perseverative_choice_rate=_mean_or_nan(early["old_prior_choice"]),
        low_contrast_trial_count=int(len(low_contrast)),
        low_contrast_new_prior_choice_rate=_mean_or_nan(low_contrast["new_prior_choice"]),
        zero_contrast_trial_count=int(len(zero_contrast)),
        zero_contrast_new_prior_choice_rate=_mean_or_nan(zero_contrast["new_prior_choice"]),
    )


def _empty_block_switch_metrics(switch_count: int = 0) -> BlockSwitchProbeMetrics:
    return BlockSwitchProbeMetrics(
        switch_count=switch_count,
        post_switch_trial_count=0,
        early_trial_count=0,
        late_trial_count=0,
        early_new_prior_choice_rate=np.nan,
        late_new_prior_choice_rate=np.nan,
        adaptation_lift=np.nan,
        early_perseverative_choice_rate=np.nan,
        low_contrast_trial_count=0,
        low_contrast_new_prior_choice_rate=np.nan,
        zero_contrast_trial_count=0,
        zero_contrast_new_prior_choice_rate=np.nan,
    )


def _block_prior_p_right(value: object) -> float:
    """Extract p(right) from a schema block-prior payload."""
    if isinstance(value, dict):
        value = value.get("p_right")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _prior_direction(p_right: float) -> str | None:
    """Return the action favored by a biased block prior."""
    if not math.isfinite(p_right) or math.isclose(p_right, 0.5):
        return None
    return "right" if p_right > 0.5 else "left"


def _mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(np.nanmean(values.astype(float)))


def _trial_success(row: pd.Series) -> bool:
    """Return trial success from `correct` when present, falling back to reward."""
    correct = row.get("correct")
    if correct is not None and pd.notna(correct):
        return bool(correct)
    reward = row.get("reward", 0.0)
    if reward is None or pd.isna(reward):
        return False
    return float(reward) > 0.0


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


def _quality_flags(metrics: dict[str, object], task: str, is_choice_only: bool = False) -> dict[str, object]:
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

    # Chrono overshoot: how far the agent's slope is from the target
    chrono_target: float | None = None
    chrono_overshoot: float | None = None
    if task == "ibl_2afc":
        # Per-session median from reference.ndjson is -44 ms/unit (high
        # variance: range -2 to -202, std ~64).  The -36 value is an
        # approximate IBL literature target retained for continuity.
        chrono_target = -36.0
    elif task == "rdm":
        chrono_target = -50.0  # Macaque RDM approximate target (ms/unit)
    if chrono_target is not None and _is_finite(effective_slope):
        chrono_overshoot = float(effective_slope) / chrono_target

    degenerate = False
    if not bias_ok:
        degenerate = True
    if not history_ok:
        degenerate = True
    if not rt_ok:
        degenerate = True
    if not chrono_ok:
        degenerate = True
    flags: dict[str, object] = {
        "bias_ok": bias_ok,
        "history_ok": history_ok,
        "rt_ok": rt_ok,
        "chronometric_ok": chrono_ok,
        "rt_ceiling_warning": ceiling_warning,
        "degenerate": degenerate,
    }
    if chrono_target is not None:
        flags["chrono_target_ms_per_unit"] = chrono_target
    if chrono_overshoot is not None:
        flags["chrono_overshoot"] = chrono_overshoot
    return flags


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
    metrics["adaptive_control_probe"] = _sanitize(asdict(compute_adaptive_control_probe_metrics(df)))
    metrics["exploration_probe"] = _sanitize(asdict(compute_exploration_probe_metrics(df)))
    metrics["block_switch_probe"] = _sanitize(asdict(compute_block_switch_probe_metrics(df)))
    metrics["quality"] = _quality_flags(metrics, task, is_choice_only)
    return _sanitize(metrics)  # type: ignore[return-value]


def load_and_compute(path: str | Path, is_choice_only: bool = False) -> dict[str, object]:
    df = load_trials(path)
    if df.empty:
        return {}
    task = df["task"].iloc[0]
    return compute_all_metrics(df, task, is_choice_only)


__all__ = [
    "AdaptiveControlProbeMetrics",
    "BlockSwitchProbeMetrics",
    "ChronometricMetrics",
    "ExplorationProbeMetrics",
    "HistoryMetrics",
    "PsychometricMetrics",
    "compute_adaptive_control_probe_metrics",
    "compute_all_metrics",
    "compute_block_switch_probe_metrics",
    "compute_chronometric",
    "compute_exploration_probe_metrics",
    "compute_history_metrics",
    "compute_psychometric",
    "load_and_compute",
    "load_trials",
]
