#!/usr/bin/env python3
"""Run matched adaptive-control validation and lesion comparisons."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import tyro

from agents.adaptive_control_agent import (
    RECOMMENDED_ADAPTIVE_CONTROL_PROFILE,
    AdaptiveControlProfile,
)


SUMMARY_METRICS: tuple[str, ...] = (
    "p_right_committed",
    "commit_rate",
    "psychometric_slope",
    "psychometric_bias",
    "lapse_low",
    "lapse_high",
    "chronometric_slope",
    "rt_range_ms",
    "ceiling_fraction",
    "win_stay",
    "lose_shift",
    "sticky_choice",
    "retry_after_failure_weak",
    "retry_after_failure_strong",
    "retry_gap",
    "switch_after_failure_weak",
    "switch_after_failure_strong",
    "switch_gap",
    "switch_after_streak_weak",
    "switch_after_streak_strong",
    "switch_after_fresh_weak",
    "switch_after_fresh_overall",
    "switch_after_stale_overall",
    "stale_switch_lift_weak",
    "stale_switch_lift_overall",
    "switch_after_unrewarded_streak_weak",
    "switch_after_unrewarded_fresh_weak",
    "unrewarded_switch_lift_weak",
    "unrewarded_streak_weak_count",
    "unrewarded_fresh_weak_count",
    "switch_after_volatile_weak",
    "switch_after_stable_weak",
    "volatile_switch_lift_weak",
    "volatile_weak_count",
    "stable_weak_count",
    "block_switch_count",
    "block_switch_adaptation_lift",
    "block_switch_early_new_prior_choice_rate",
    "block_switch_late_new_prior_choice_rate",
    "block_switch_early_perseverative_choice_rate",
    "block_switch_low_contrast_new_prior_choice_rate",
    "block_switch_zero_contrast_new_prior_choice_rate",
    "weak_streak_count",
    "strong_streak_count",
    "fresh_weak_count",
    "fresh_count",
    "stale_count",
    "exploration_gap",
)

PAIRED_METRICS: tuple[str, ...] = (
    "psychometric_slope",
    "chronometric_slope",
    "rt_range_ms",
    "win_stay",
    "lose_shift",
    "sticky_choice",
    "retry_gap",
    "switch_gap",
    "switch_after_streak_weak",
    "switch_after_fresh_weak",
    "stale_switch_lift_weak",
    "stale_switch_lift_overall",
    "unrewarded_switch_lift_weak",
    "volatile_switch_lift_weak",
    "switch_after_unrewarded_streak_weak",
    "switch_after_unrewarded_fresh_weak",
    "switch_after_volatile_weak",
    "switch_after_stable_weak",
    "block_switch_adaptation_lift",
    "block_switch_early_new_prior_choice_rate",
    "block_switch_late_new_prior_choice_rate",
    "block_switch_early_perseverative_choice_rate",
    "block_switch_low_contrast_new_prior_choice_rate",
    "block_switch_zero_contrast_new_prior_choice_rate",
    "exploration_gap",
)

COUNT_KEYS: tuple[str, ...] = (
    "bias_ok",
    "history_ok",
    "rt_ok",
    "chronometric_ok",
    "rt_ceiling_warning",
    "degenerate",
)


@dataclass(frozen=True, slots=True)
class ValidationCondition:
    """One validation condition in the matched adaptive-control suite."""

    label: str
    description: str
    control_profile: AdaptiveControlProfile
    extra_args: tuple[str, ...] = ()
    control_uncertainty_power: float | None = None
    persistence_bias_scale: float | None = None
    exploration_bias_scale: float | None = None


@dataclass(slots=True)
class ValidationSuiteArgs:
    """Arguments for the adaptive-control matched validation suite."""

    run_root: Path = Path("runs/adaptive_control_validation_suite")
    seeds: Sequence[int] = (42, 123, 456, 789, 2026)
    task: Literal["ibl_2afc", "rdm"] = "ibl_2afc"
    episodes: int = 5
    trials_per_episode: int = 400
    epochs: int = 3
    hidden_size: int = 64
    learning_rate: float = 1e-3
    max_sessions: int = 20
    max_trials_per_session: int = 128
    min_commit_steps: int = 5
    max_commit_steps: int = 300
    drift_scale: float = 6.0
    history_bias_scale: float = 2.0
    history_drift_scale: float = 0.3
    lapse_rate: float = 0.05
    persistence_bias_scale: float = 1.6
    exploration_bias_scale: float = 0.8
    persistence_learning_rate: float = 0.8
    control_uncertainty_power: float = 2.0
    include_exploration_only: bool = True
    include_gate_lesion: bool = False
    gate_lesion_uncertainty_power: float = 1.0
    skip_existing: bool = True
    dry_run: bool = False

    def run(self) -> None:
        """Train, evaluate, and summarize the matched validation suite."""
        self.run_root.mkdir(parents=True, exist_ok=True)
        conditions = self._conditions()

        for condition in conditions:
            for seed in self.seeds:
                run_dir = self.run_root / f"{condition.label}_seed{seed}"
                metrics_path = run_dir / "metrics.json"
                if self.skip_existing and metrics_path.exists():
                    print(f"[SKIP] {condition.label} seed={seed}: metrics already exist")
                    continue
                print(f"\n{'=' * 80}")
                print(f"Running {condition.label} | seed={seed}")
                print(condition.description)
                print(f"Output: {run_dir}")
                print(f"{'=' * 80}")
                self._run_command(self._build_train_command(run_dir, seed, condition))
                self._run_command(self._build_eval_command(run_dir))

        if self.dry_run:
            return

        per_run_rows = self._collect_per_run_rows(conditions)
        if not per_run_rows:
            print("No completed metrics found; nothing to summarize.")
            return

        aggregate_rows = _aggregate_rows(per_run_rows, conditions)
        paired_rows = _paired_delta_rows(per_run_rows, baseline_condition="true_no_control")
        paired_summary_rows = _paired_delta_summary_rows(paired_rows)

        _write_csv(self.run_root / "per_run_comparison.csv", per_run_rows)
        _write_csv(self.run_root / "aggregate_summary.csv", aggregate_rows)
        _write_csv(self.run_root / "paired_deltas.csv", paired_rows)
        _write_csv(self.run_root / "paired_delta_summary.csv", paired_summary_rows)
        self._write_json_summary(conditions, aggregate_rows, paired_summary_rows)
        self._print_summary(aggregate_rows, paired_summary_rows)

    def _conditions(self) -> list[ValidationCondition]:
        conditions = [
            ValidationCondition(
                label="true_no_control",
                description="Clean lesion: disables all adaptive-control fast state and overlays.",
                control_profile="no_control",
            ),
            ValidationCondition(
                label="exploration_only",
                description="Experimental exploration lesion; persistence disabled and exploration unvalidated.",
                control_profile="exploration_only",
            ),
            ValidationCondition(
                label="persistence_only",
                description="Recommended validated claim: persistence/retry enabled; exploration disabled.",
                control_profile="persistence_only",
            ),
            ValidationCondition(
                label="full_control",
                description="Comparison condition: persistence plus experimental exploration.",
                control_profile="full_control",
            ),
        ]
        if self.include_gate_lesion:
            conditions.append(
                ValidationCondition(
                    label="linear_gate_full_control",
                    description="Gate lesion: weakens nonlinear uncertainty gating while leaving control enabled.",
                    control_profile="full_control",
                    control_uncertainty_power=self.gate_lesion_uncertainty_power,
                )
            )
        if not self.include_exploration_only:
            conditions = [condition for condition in conditions if condition.label != "exploration_only"]
        return conditions

    def _build_train_command(
        self,
        run_dir: Path,
        seed: int,
        condition: ValidationCondition,
    ) -> list[str]:
        cmd = [
            sys.executable,
            "scripts/train_adaptive_control.py",
            "--output-dir",
            str(run_dir),
            "--task",
            self.task,
            "--seed",
            str(seed),
            "--episodes",
            str(self.episodes),
            "--trials-per-episode",
            str(self.trials_per_episode),
            "--epochs",
            str(self.epochs),
            "--hidden-size",
            str(self.hidden_size),
            "--learning-rate",
            str(self.learning_rate),
            "--control-profile",
            condition.control_profile,
            "--max-sessions",
            str(self.max_sessions),
            "--max-trials-per-session",
            str(self.max_trials_per_session),
            "--min-commit-steps",
            str(self.min_commit_steps),
            "--max-commit-steps",
            str(self.max_commit_steps),
            "--drift-scale",
            str(self.drift_scale),
            "--history-bias-scale",
            str(self.history_bias_scale),
            "--history-drift-scale",
            str(self.history_drift_scale),
            "--lapse-rate",
            str(self.lapse_rate),
            "--persistence-bias-scale",
            str(self._condition_persistence_bias_scale(condition)),
            "--exploration-bias-scale",
            str(self._condition_exploration_bias_scale(condition)),
            "--persistence-learning-rate",
            str(self.persistence_learning_rate),
            "--control-uncertainty-power",
            str(self._condition_control_uncertainty_power(condition)),
            *condition.extra_args,
        ]
        return cmd

    def _condition_persistence_bias_scale(self, condition: ValidationCondition) -> float:
        """Return the resolved persistence scale for one condition."""
        if condition.persistence_bias_scale is not None:
            return condition.persistence_bias_scale
        return self.persistence_bias_scale

    def _condition_exploration_bias_scale(self, condition: ValidationCondition) -> float:
        """Return the resolved exploration scale for one condition."""
        if condition.exploration_bias_scale is not None:
            return condition.exploration_bias_scale
        return self.exploration_bias_scale

    def _condition_control_uncertainty_power(self, condition: ValidationCondition) -> float:
        """Return the resolved uncertainty-gate power for one condition."""
        if condition.control_uncertainty_power is not None:
            return condition.control_uncertainty_power
        return self.control_uncertainty_power

    def _condition_payload(self, condition: ValidationCondition) -> dict[str, object]:
        """Return reproducible condition settings for summaries."""
        return {
            "label": condition.label,
            "description": condition.description,
            "control_profile": condition.control_profile,
            "persistence_bias_scale": self._condition_persistence_bias_scale(condition),
            "exploration_bias_scale": self._condition_exploration_bias_scale(condition),
            "control_uncertainty_power": self._condition_control_uncertainty_power(condition),
            "extra_args": list(condition.extra_args),
        }

    @staticmethod
    def _build_eval_command(run_dir: Path) -> list[str]:
        return [
            sys.executable,
            "scripts/evaluate_agent.py",
            "--run",
            str(run_dir),
        ]

    def _run_command(self, cmd: list[str]) -> None:
        print("[CMD]", " ".join(cmd))
        if self.dry_run:
            return
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")

    def _collect_per_run_rows(self, conditions: Sequence[ValidationCondition]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        condition_by_label = {condition.label: condition for condition in conditions}
        for condition in conditions:
            for seed in self.seeds:
                run_dir = self.run_root / f"{condition.label}_seed{seed}"
                metrics = _load_metrics(run_dir)
                if metrics is None:
                    print(f"[MISSING] {condition.label} seed={seed}: {run_dir / 'metrics.json'}")
                    continue
                row = _summarize_run(metrics)
                row.update(
                    {
                        "condition": condition.label,
                        "description": condition_by_label[condition.label].description,
                        "control_profile": condition.control_profile,
                        "persistence_bias_scale": self._condition_persistence_bias_scale(condition),
                        "exploration_bias_scale": self._condition_exploration_bias_scale(condition),
                        "control_uncertainty_power": self._condition_control_uncertainty_power(condition),
                        "seed": seed,
                        "run_dir": str(run_dir),
                    }
                )
                rows.append(row)
        rows.sort(key=lambda row: (str(row["condition"]), int(row["seed"])))
        return rows

    def _write_json_summary(
        self,
        conditions: Sequence[ValidationCondition],
        aggregate_rows: list[dict[str, object]],
        paired_summary_rows: list[dict[str, object]],
    ) -> None:
        payload = {
            "config": {
                "seeds": list(self.seeds),
                "episodes": self.episodes,
                "epochs": self.epochs,
                "max_sessions": self.max_sessions,
                "max_trials_per_session": self.max_trials_per_session,
                "drift_scale": self.drift_scale,
                "persistence_bias_scale": self.persistence_bias_scale,
                "exploration_bias_scale": self.exploration_bias_scale,
                "control_uncertainty_power": self.control_uncertainty_power,
                "recommended_control_profile": RECOMMENDED_ADAPTIVE_CONTROL_PROFILE,
                "include_exploration_only": self.include_exploration_only,
                "include_gate_lesion": self.include_gate_lesion,
            },
            "conditions": [self._condition_payload(condition) for condition in conditions],
            "aggregate": aggregate_rows,
            "paired_delta_summary": paired_summary_rows,
        }
        summary_path = self.run_root / "validation_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _print_summary(
        self,
        aggregate_rows: list[dict[str, object]],
        paired_summary_rows: list[dict[str, object]],
    ) -> None:
        print(f"\n{'=' * 80}")
        print("Adaptive Control Validation Suite")
        print(f"{'=' * 80}")
        print(
            f"{'condition':<26} | {'n':>2} | {'psych':>8} | {'chrono':>8} | "
            f"{'retry':>7} | {'stale':>7} | {'ceil':>4} | {'deg':>3}"
        )
        print("-" * 92)
        for row in aggregate_rows:
            print(
                f"{str(row['condition']):<26} | "
                f"{int(row['num_seeds']):>2} | "
                f"{_fmt(row.get('psychometric_slope_mean')):>8} | "
                f"{_fmt(row.get('chronometric_slope_mean')):>8} | "
                f"{_fmt(row.get('retry_gap_mean'), decimals=3):>7} | "
                f"{_fmt(row.get('stale_switch_lift_weak_mean'), decimals=3):>7} | "
                f"{int(row.get('rt_ceiling_warning_count', 0)):>4} | "
                f"{int(row.get('degenerate_count', 0)):>3}"
            )

        if paired_summary_rows:
            print(
                f"\n{'comparison':<42} | {'d_retry':>8} | {'pos':>5} | "
                f"{'d_stale':>8} | {'d_psych':>8} | {'d_chrono':>8}"
            )
            print("-" * 92)
            for row in paired_summary_rows:
                comparison = str(row["comparison"])
                print(
                    f"{comparison:<42} | "
                    f"{_fmt(row.get('delta_retry_gap_mean'), decimals=3):>8} | "
                    f"{int(row.get('delta_retry_gap_positive_count', 0)):>5} | "
                    f"{_fmt(row.get('delta_stale_switch_lift_weak_mean'), decimals=3):>8} | "
                    f"{_fmt(row.get('delta_psychometric_slope_mean')):>8} | "
                    f"{_fmt(row.get('delta_chronometric_slope_mean')):>8}"
                )

            print(
                f"\n{'comparison':<42} | {'d_unrew':>8} | {'pos':>5} | "
                f"{'d_vol':>8} | {'pos':>5}"
            )
            print("-" * 76)
            for row in paired_summary_rows:
                comparison = str(row["comparison"])
                print(
                    f"{comparison:<42} | "
                    f"{_fmt(row.get('delta_unrewarded_switch_lift_weak_mean'), decimals=3):>8} | "
                    f"{int(row.get('delta_unrewarded_switch_lift_weak_positive_count', 0)):>5} | "
                    f"{_fmt(row.get('delta_volatile_switch_lift_weak_mean'), decimals=3):>8} | "
                    f"{int(row.get('delta_volatile_switch_lift_weak_positive_count', 0)):>5}"
                )

            print(
                f"\n{'comparison':<42} | {'d_block':>8} | {'pos':>5} | "
                f"{'d_early':>8} | {'d_late':>8}"
            )
            print("-" * 83)
            for row in paired_summary_rows:
                comparison = str(row["comparison"])
                print(
                    f"{comparison:<42} | "
                    f"{_fmt(row.get('delta_block_switch_adaptation_lift_mean'), decimals=3):>8} | "
                    f"{int(row.get('delta_block_switch_adaptation_lift_positive_count', 0)):>5} | "
                    f"{_fmt(row.get('delta_block_switch_early_new_prior_choice_rate_mean'), decimals=3):>8} | "
                    f"{_fmt(row.get('delta_block_switch_late_new_prior_choice_rate_mean'), decimals=3):>8}"
                )

        print(f"\nPer-run summary saved to {self.run_root / 'per_run_comparison.csv'}")
        print(f"Aggregate summary saved to {self.run_root / 'aggregate_summary.csv'}")
        print(f"Paired deltas saved to {self.run_root / 'paired_deltas.csv'}")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_metrics(run_dir: Path) -> dict[str, Any] | None:
    payload = _load_json(run_dir / "metrics.json")
    if payload is None:
        return None
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        return None
    return metrics


def _summarize_run(metrics: dict[str, Any]) -> dict[str, object]:
    psychometric = _as_dict(metrics.get("psychometric"))
    chronometric = _as_dict(metrics.get("chronometric"))
    history = _as_dict(metrics.get("history"))
    probe = _as_dict(metrics.get("adaptive_control_probe"))
    exploration_probe = _as_dict(metrics.get("exploration_probe"))
    block_switch_probe = _as_dict(metrics.get("block_switch_probe"))
    quality = _as_dict(metrics.get("quality"))

    row: dict[str, object] = {
        "p_right_overall": metrics.get("p_right_overall"),
        "p_right_committed": metrics.get("p_right_committed"),
        "commit_rate": metrics.get("commit_rate"),
        "psychometric_slope": psychometric.get("slope"),
        "psychometric_bias": psychometric.get("bias"),
        "lapse_low": psychometric.get("lapse_low"),
        "lapse_high": psychometric.get("lapse_high"),
        "chronometric_slope": chronometric.get("slope_ms_per_unit"),
        "chronometric_intercept_ms": chronometric.get("intercept_ms"),
        "ceiling_fraction": chronometric.get("ceiling_fraction"),
        "rt_range_ms": chronometric.get("rt_range_ms"),
        "win_stay": history.get("win_stay"),
        "lose_shift": history.get("lose_shift"),
        "sticky_choice": history.get("sticky_choice"),
        "prev_choice_beta": history.get("prev_choice_beta"),
        "prev_correct_beta": history.get("prev_correct_beta"),
        "retry_after_failure_weak": probe.get("retry_after_failure_weak"),
        "retry_after_failure_strong": probe.get("retry_after_failure_strong"),
        "switch_after_failure_weak": probe.get("switch_after_failure_weak"),
        "switch_after_failure_strong": probe.get("switch_after_failure_strong"),
        "weak_failure_count": probe.get("weak_failure_count"),
        "strong_failure_count": probe.get("strong_failure_count"),
        "switch_after_streak_weak": exploration_probe.get("switch_after_streak_weak"),
        "switch_after_streak_strong": exploration_probe.get("switch_after_streak_strong"),
        "switch_after_fresh_weak": exploration_probe.get("switch_after_fresh_weak"),
        "switch_after_fresh_overall": exploration_probe.get("switch_after_fresh_overall"),
        "switch_after_stale_overall": exploration_probe.get("switch_after_stale_overall"),
        "stale_switch_lift_weak": exploration_probe.get("stale_switch_lift_weak"),
        "stale_switch_lift_overall": exploration_probe.get("stale_switch_lift_overall"),
        "switch_after_unrewarded_streak_weak": exploration_probe.get("switch_after_unrewarded_streak_weak"),
        "switch_after_unrewarded_fresh_weak": exploration_probe.get("switch_after_unrewarded_fresh_weak"),
        "unrewarded_switch_lift_weak": exploration_probe.get("unrewarded_switch_lift_weak"),
        "unrewarded_streak_weak_count": exploration_probe.get("unrewarded_streak_weak_count"),
        "unrewarded_fresh_weak_count": exploration_probe.get("unrewarded_fresh_weak_count"),
        "switch_after_volatile_weak": exploration_probe.get("switch_after_volatile_weak"),
        "switch_after_stable_weak": exploration_probe.get("switch_after_stable_weak"),
        "volatile_switch_lift_weak": exploration_probe.get("volatile_switch_lift_weak"),
        "volatile_weak_count": exploration_probe.get("volatile_weak_count"),
        "stable_weak_count": exploration_probe.get("stable_weak_count"),
        "block_switch_count": block_switch_probe.get("switch_count"),
        "block_switch_post_switch_trial_count": block_switch_probe.get("post_switch_trial_count"),
        "block_switch_early_trial_count": block_switch_probe.get("early_trial_count"),
        "block_switch_late_trial_count": block_switch_probe.get("late_trial_count"),
        "block_switch_adaptation_lift": block_switch_probe.get("adaptation_lift"),
        "block_switch_early_new_prior_choice_rate": block_switch_probe.get("early_new_prior_choice_rate"),
        "block_switch_late_new_prior_choice_rate": block_switch_probe.get("late_new_prior_choice_rate"),
        "block_switch_early_perseverative_choice_rate": block_switch_probe.get(
            "early_perseverative_choice_rate",
        ),
        "block_switch_low_contrast_new_prior_choice_rate": block_switch_probe.get(
            "low_contrast_new_prior_choice_rate",
        ),
        "block_switch_zero_contrast_new_prior_choice_rate": block_switch_probe.get(
            "zero_contrast_new_prior_choice_rate",
        ),
        "weak_streak_count": exploration_probe.get("weak_streak_count"),
        "strong_streak_count": exploration_probe.get("strong_streak_count"),
        "fresh_weak_count": exploration_probe.get("fresh_weak_count"),
        "fresh_count": exploration_probe.get("fresh_count"),
        "stale_count": exploration_probe.get("stale_count"),
        "bias_ok": quality.get("bias_ok"),
        "history_ok": quality.get("history_ok"),
        "rt_ok": quality.get("rt_ok"),
        "chronometric_ok": quality.get("chronometric_ok"),
        "rt_ceiling_warning": quality.get("rt_ceiling_warning"),
        "degenerate": quality.get("degenerate"),
    }
    _add_gap(row, "retry_gap", "retry_after_failure_weak", "retry_after_failure_strong")
    _add_gap(row, "switch_gap", "switch_after_failure_strong", "switch_after_failure_weak")
    _add_gap(row, "exploration_gap", "switch_after_streak_weak", "switch_after_streak_strong")
    return row


def _aggregate_rows(
    per_run_rows: Sequence[dict[str, object]],
    conditions: Sequence[ValidationCondition],
) -> list[dict[str, object]]:
    rows_by_condition: dict[str, list[dict[str, object]]] = {}
    for row in per_run_rows:
        rows_by_condition.setdefault(str(row["condition"]), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for condition in conditions:
        rows = rows_by_condition.get(condition.label, [])
        if not rows:
            continue
        aggregate: dict[str, object] = {
            "condition": condition.label,
            "description": condition.description,
            "num_seeds": len(rows),
        }
        for key in COUNT_KEYS:
            aggregate[f"{key}_count"] = sum(1 for row in rows if row.get(key) is True)
        for key in SUMMARY_METRICS:
            values = [_as_float(row.get(key)) for row in rows]
            finite_values = [value for value in values if value is not None]
            if finite_values:
                aggregate[f"{key}_mean"] = _mean(finite_values)
                aggregate[f"{key}_std"] = _std(finite_values)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _paired_delta_rows(
    per_run_rows: Sequence[dict[str, object]],
    *,
    baseline_condition: str,
) -> list[dict[str, object]]:
    by_condition_seed: dict[tuple[str, int], dict[str, object]] = {}
    for row in per_run_rows:
        seed = row.get("seed")
        if not isinstance(seed, int):
            continue
        by_condition_seed[(str(row["condition"]), seed)] = row

    paired_rows: list[dict[str, object]] = []
    for row in per_run_rows:
        condition = str(row["condition"])
        if condition == baseline_condition:
            continue
        seed = row.get("seed")
        if not isinstance(seed, int):
            continue
        baseline = by_condition_seed.get((baseline_condition, seed))
        if baseline is None:
            continue
        comparison = f"{condition}_minus_{baseline_condition}"
        delta_row: dict[str, object] = {
            "comparison": comparison,
            "condition": condition,
            "baseline_condition": baseline_condition,
            "seed": seed,
        }
        for key in PAIRED_METRICS:
            value = _as_float(row.get(key))
            baseline_value = _as_float(baseline.get(key))
            if value is not None and baseline_value is not None:
                delta_row[f"delta_{key}"] = value - baseline_value
        paired_rows.append(delta_row)
    paired_rows.sort(key=lambda row: (str(row["comparison"]), int(row["seed"])))
    return paired_rows


def _paired_delta_summary_rows(paired_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_comparison: dict[str, list[dict[str, object]]] = {}
    for row in paired_rows:
        rows_by_comparison.setdefault(str(row["comparison"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for comparison, rows in sorted(rows_by_comparison.items()):
        summary: dict[str, object] = {
            "comparison": comparison,
            "num_seeds": len(rows),
        }
        for key in PAIRED_METRICS:
            delta_key = f"delta_{key}"
            values = [_as_float(row.get(delta_key)) for row in rows]
            finite_values = [value for value in values if value is not None]
            if finite_values:
                summary[f"{delta_key}_mean"] = _mean(finite_values)
                summary[f"{delta_key}_std"] = _std(finite_values)
                summary[f"{delta_key}_positive_count"] = sum(1 for value in finite_values if value > 0.0)
        summary_rows.append(summary)
    return summary_rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _add_gap(row: dict[str, object], output_key: str, high_key: str, low_key: str) -> None:
    high = _as_float(row.get(high_key))
    low = _as_float(row.get(low_key))
    if high is not None and low is not None:
        row[output_key] = high - low


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _fmt(value: object, *, decimals: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{decimals}f}"


def main() -> None:
    """CLI entry point."""
    tyro.cli(ValidationSuiteArgs).run()


if __name__ == "__main__":
    main()
