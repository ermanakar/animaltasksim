from __future__ import annotations

from pathlib import Path

import pytest

from scripts.adaptive_control_validation_suite import (
    ValidationSuiteArgs,
    _aggregate_rows,
    _paired_delta_rows,
    _paired_delta_summary_rows,
)


def test_validation_suite_builds_clean_no_control_lesion_command(tmp_path: Path) -> None:
    args = ValidationSuiteArgs(run_root=tmp_path)
    condition = args._conditions()[0]

    cmd = args._build_train_command(tmp_path / "run", 42, condition)

    assert condition.label == "true_no_control"
    assert "--no-control-state-enabled" in cmd
    assert "--no-persistence-enabled" in cmd
    assert "--no-exploration-enabled" in cmd
    assert cmd[cmd.index("--drift-scale") + 1] == "6.0"
    assert cmd[cmd.index("--persistence-bias-scale") + 1] == "1.6"
    assert cmd[cmd.index("--control-uncertainty-power") + 1] == "2.0"


def test_validation_suite_can_include_gate_lesion(tmp_path: Path) -> None:
    args = ValidationSuiteArgs(
        run_root=tmp_path,
        include_gate_lesion=True,
        gate_lesion_uncertainty_power=1.0,
    )

    conditions = args._conditions()
    gate_lesion = conditions[-1]
    cmd = args._build_train_command(tmp_path / "run", 42, gate_lesion)

    assert gate_lesion.label == "linear_gate_full_control"
    assert cmd.count("--control-uncertainty-power") == 1
    assert cmd[cmd.index("--control-uncertainty-power") + 1] == "1.0"


def test_validation_suite_builds_exploration_only_condition(tmp_path: Path) -> None:
    args = ValidationSuiteArgs(run_root=tmp_path)
    condition = next(condition for condition in args._conditions() if condition.label == "exploration_only")

    cmd = args._build_train_command(tmp_path / "run", 42, condition)

    assert "--no-persistence-enabled" in cmd
    assert "--no-exploration-enabled" not in cmd
    assert "--no-control-state-enabled" not in cmd


def test_validation_suite_paired_delta_summary() -> None:
    conditions = ValidationSuiteArgs()._conditions()
    rows = [
        {
            "condition": "true_no_control",
            "seed": 42,
            "psychometric_slope": 25.0,
            "chronometric_slope": -40.0,
            "retry_gap": 0.10,
            "stale_switch_lift_weak": 0.02,
            "degenerate": False,
            "chronometric_ok": True,
        },
        {
            "condition": "true_no_control",
            "seed": 123,
            "psychometric_slope": 27.0,
            "chronometric_slope": -42.0,
            "retry_gap": 0.12,
            "stale_switch_lift_weak": 0.03,
            "degenerate": False,
            "chronometric_ok": True,
        },
        {
            "condition": "full_control",
            "seed": 42,
            "psychometric_slope": 22.0,
            "chronometric_slope": -36.0,
            "retry_gap": 0.16,
            "stale_switch_lift_weak": 0.08,
            "degenerate": False,
            "chronometric_ok": True,
        },
        {
            "condition": "full_control",
            "seed": 123,
            "psychometric_slope": 23.0,
            "chronometric_slope": -35.0,
            "retry_gap": 0.17,
            "stale_switch_lift_weak": 0.09,
            "degenerate": False,
            "chronometric_ok": True,
        },
    ]

    aggregate = _aggregate_rows(rows, conditions)
    paired = _paired_delta_rows(rows, baseline_condition="true_no_control")
    summary = _paired_delta_summary_rows(paired)

    full_control = next(row for row in aggregate if row["condition"] == "full_control")
    full_control_delta = next(row for row in summary if row["comparison"] == "full_control_minus_true_no_control")

    assert full_control["num_seeds"] == 2
    assert full_control["retry_gap_mean"] == pytest.approx(0.165)
    assert full_control["stale_switch_lift_weak_mean"] == pytest.approx(0.085)
    assert full_control_delta["delta_retry_gap_mean"] == pytest.approx(0.055)
    assert full_control_delta["delta_retry_gap_positive_count"] == 2
    assert full_control_delta["delta_stale_switch_lift_weak_mean"] == pytest.approx(0.06)
    assert full_control_delta["delta_stale_switch_lift_weak_positive_count"] == 2
    assert full_control_delta["delta_psychometric_slope_mean"] == pytest.approx(-3.5)
