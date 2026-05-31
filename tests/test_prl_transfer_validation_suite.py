from __future__ import annotations

from pathlib import Path

from scripts.prl_transfer_validation_suite import PRLTransferSuiteArgs


def test_prl_transfer_suite_builds_transfer_conditions(tmp_path: Path) -> None:
    args = PRLTransferSuiteArgs(run_root=tmp_path)

    conditions = args._conditions()
    candidate = conditions[-1]
    cmd = args._build_train_command(tmp_path / "run", 42, candidate)

    assert args.task == "prl"
    assert [condition.label for condition in conditions] == [
        "true_no_control",
        "exploration_only",
        "persistence_only",
        "full_control_default",
        "full_control_persist_half",
    ]
    assert candidate.persistence_bias_scale == 0.8
    assert cmd[cmd.index("--task") + 1] == "prl"
    assert cmd[cmd.index("--persistence-bias-scale") + 1] == "0.8"


def test_prl_transfer_suite_prints_hidden_reversal_scorecard(tmp_path: Path, capsys) -> None:
    args = PRLTransferSuiteArgs(run_root=tmp_path)

    args._print_summary(
        aggregate_rows=[
            {
                "condition": "persistence_only",
                "num_seeds": 1,
                "prl_optimal_choice_rate_mean": 0.6,
                "prl_reward_rate_mean": 0.55,
                "prl_early_optimal_choice_rate_mean": 0.4,
                "prl_late_optimal_choice_rate_mean": 0.7,
                "prl_adaptation_lift_mean": 0.3,
                "prl_end_block_optimal_choice_rate_mean": 0.8,
                "prl_block_learning_lift_mean": 0.4,
                "prl_reversal_count_mean": 2.0,
                "reversal_probe_ok_count": 1,
                "degenerate_count": 0,
            }
        ],
        paired_summary_rows=[],
    )

    output = capsys.readouterr().out
    assert "Adaptive Control PRL Transfer Suite" in output
    assert "optimal" in output
    assert "learn" in output
