from __future__ import annotations

import json
from pathlib import Path

from animaltasksim.registry import ExperimentRegistry, extract_metadata_from_run


def test_registry_extracts_prl_adaptive_control_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "prl_adaptive_control_smoke"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "task": "prl",
                "seed": 42,
                "control_state_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "prl": {
                        "optimal_choice_rate": 0.7,
                        "reward_rate": 0.65,
                        "adaptation_lift": 0.3,
                        "block_learning_lift": 0.4,
                    },
                    "quality": {
                        "reversal_probe_ok": True,
                        "degenerate": False,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trials.ndjson").touch()

    metadata = extract_metadata_from_run(run_dir)

    assert metadata is not None
    assert metadata.task == "prl"
    assert metadata.agent == "adaptive_control"
    assert metadata.prl_optimal_choice_rate == 0.7
    assert metadata.prl_reward_rate == 0.65
    assert metadata.prl_adaptation_lift == 0.3
    assert metadata.prl_block_learning_lift == 0.4
    assert metadata.quality == {
        "reversal_probe_ok": True,
        "degenerate": False,
    }

    registry = ExperimentRegistry(tmp_path / "runs" / "registry.json")
    registry.add(metadata)

    assert registry.get(run_dir.name) == metadata


def test_explicit_config_task_wins_over_directory_name(tmp_path: Path) -> None:
    # An IBL control run that happens to live inside a PRL-named suite directory
    # must not be reclassified as a PRL run: the config task is authoritative.
    run_dir = tmp_path / "runs" / "prl_transfer_suite_ibl_control_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"task": "ibl_2afc", "seed": 42, "control_state_enabled": True}),
        encoding="utf-8",
    )

    metadata = extract_metadata_from_run(run_dir)

    assert metadata is not None
    assert metadata.task == "ibl_2afc"


def test_directory_name_infers_task_when_config_task_missing(tmp_path: Path) -> None:
    # With no task/env in the config, the directory-name heuristic still applies.
    run_dir = tmp_path / "runs" / "prl_arbitration_diagnostic_v9"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"seed": 42, "control_state_enabled": True}),
        encoding="utf-8",
    )

    metadata = extract_metadata_from_run(run_dir)

    assert metadata is not None
    assert metadata.task == "prl"
