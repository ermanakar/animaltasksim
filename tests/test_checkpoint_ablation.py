"""Check acute-ablation configuration and historical-output protection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.checkpoint_ablation import CheckpointAblationArgs


def test_checkpoint_config_preserves_source_and_disables_optimization(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = {"task": "prl", "seed": 123, "epochs": 5, "hidden_size": 32,
               "episodes": 4, "reference_log": "old.ndjson", "active_control_profile": "full_control"}
    (source / "config.json").write_text(json.dumps(payload))
    args = CheckpointAblationArgs(source, tmp_path / "out", reference_log=tmp_path / "new.ndjson")
    config = args._config(tmp_path / "out" / "no_control", "no_control")
    assert config.epochs == 0
    assert config.hidden_size == 32
    assert config.seed == 123
    assert config.episodes == 4
    assert config.reference_log == tmp_path / "new.ndjson"
    assert config.active_control_profile == "no_control"
    assert json.loads((source / "config.json").read_text()) == payload


def test_checkpoint_ablation_refuses_existing_results(tmp_path: Path) -> None:
    (tmp_path / "result").write_text("preserve")
    args = CheckpointAblationArgs(tmp_path / "missing_source", tmp_path)
    with pytest.raises(RuntimeError, match="new, empty"):
        args.run()
    assert (tmp_path / "result").read_text() == "preserve"
