from __future__ import annotations

from pathlib import Path

from scripts.adaptive_control_interaction_sweep import InteractionSweepArgs


def test_interaction_sweep_defines_baselines_and_arbitration_profiles(tmp_path: Path) -> None:
    args = InteractionSweepArgs(run_root=tmp_path)

    conditions = args._conditions()
    labels = [condition.label for condition in conditions]
    by_label = {condition.label: condition for condition in conditions}

    assert labels == [
        "true_no_control",
        "exploration_only",
        "persistence_only",
        "full_control_default",
        "full_control_persist_half",
        "full_control_persist_quarter",
        "full_control_explore_strong",
        "full_control_explore_double",
        "full_control_balanced",
        "full_control_explore_dominant",
    ]
    assert by_label["full_control_default"].persistence_bias_scale == 1.6
    assert by_label["full_control_default"].exploration_bias_scale == 0.8
    assert by_label["full_control_persist_half"].persistence_bias_scale == 0.8
    assert by_label["full_control_persist_half"].exploration_bias_scale == 0.8
    assert by_label["full_control_explore_dominant"].persistence_bias_scale == 0.4
    assert by_label["full_control_explore_dominant"].exploration_bias_scale == 1.6


def test_interaction_sweep_passes_condition_scale_overrides(tmp_path: Path) -> None:
    args = InteractionSweepArgs(run_root=tmp_path)
    condition = next(
        condition for condition in args._conditions() if condition.label == "full_control_persist_half"
    )

    cmd = args._build_train_command(tmp_path / "run", 42, condition)

    assert cmd[cmd.index("--control-profile") + 1] == "full_control"
    assert cmd[cmd.index("--persistence-bias-scale") + 1] == "0.8"
    assert cmd[cmd.index("--exploration-bias-scale") + 1] == "0.8"
