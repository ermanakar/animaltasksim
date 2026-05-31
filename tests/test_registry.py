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
