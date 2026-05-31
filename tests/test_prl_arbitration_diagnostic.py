from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agents.adaptive_control_config import AdaptiveControlConfig
from scripts.prl_arbitration_diagnostic import (
    DIAGNOSTIC_FILENAME,
    PRLArbitrationDiagnosticArgs,
    _load_source_config,
    _reroll_is_complete,
    _summarize_reversal_windows,
)


def _write_run(run_dir: Path, sidecar_lines: list[str], manifest_rows: int | None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    sidecar = run_dir / DIAGNOSTIC_FILENAME
    sidecar.write_text("".join(f"{line}\n" for line in sidecar_lines), encoding="utf-8")
    if manifest_rows is not None:
        (run_dir / "diagnostic_manifest.json").write_text(
            json.dumps({"diagnostic_rows": manifest_rows}), encoding="utf-8"
        )
    return sidecar


def test_reroll_complete_when_manifest_rows_match(tmp_path: Path) -> None:
    sidecar = _write_run(tmp_path, ['{"a": 1}', '{"a": 2}'], manifest_rows=2)
    assert _reroll_is_complete(tmp_path, sidecar) is True


def test_reroll_incomplete_when_sidecar_truncated(tmp_path: Path) -> None:
    sidecar = _write_run(tmp_path, ['{"a": 1}'], manifest_rows=2)
    assert _reroll_is_complete(tmp_path, sidecar) is False


def test_reroll_incomplete_on_malformed_sidecar_without_raising(tmp_path: Path) -> None:
    sidecar = _write_run(tmp_path, ['{"a": 1}', "{not valid json"], manifest_rows=2)
    assert _reroll_is_complete(tmp_path, sidecar) is False


def test_reroll_incomplete_when_manifest_missing(tmp_path: Path) -> None:
    sidecar = _write_run(tmp_path, ['{"a": 1}', '{"a": 2}'], manifest_rows=None)
    assert _reroll_is_complete(tmp_path, sidecar) is False


def test_reroll_incomplete_when_sidecar_missing(tmp_path: Path) -> None:
    # Stale manifest left behind without its sidecar must not count as complete.
    (tmp_path / "diagnostic_manifest.json").write_text(
        json.dumps({"diagnostic_rows": 2}), encoding="utf-8"
    )
    assert _reroll_is_complete(tmp_path, tmp_path / DIAGNOSTIC_FILENAME) is False


def test_prl_diagnostic_rebuilds_saved_source_config(tmp_path: Path) -> None:
    source_config = AdaptiveControlConfig(
        task="prl",
        output_dir=tmp_path / "source",
        episodes=4,
        trials_per_episode=400,
    )
    payload = asdict(source_config)
    payload["active_control_profile"] = source_config.active_control_profile
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload, default=str), encoding="utf-8")

    config = _load_source_config(
        config_path,
        output_run=tmp_path / "diagnostic",
        episodes=1,
        trials_per_episode=12,
    )

    assert config.task == "prl"
    assert config.output_dir == tmp_path / "diagnostic"
    assert config.epochs == 0
    assert config.episodes == 1
    assert config.trials_per_episode == 12


def test_prl_diagnostic_summarizes_post_reversal_windows() -> None:
    records = [
        {
            "session_id": "s1",
            "trial_index": trial_index,
            "reversal": trial_index == 2,
            "correct": trial_index >= 4,
            "reward": 1.0 if trial_index >= 4 else -0.1,
            "base_stay_tendency": 0.0,
            "raw_control_bias": 0.1,
            "control_bias": 0.1,
            "control_gate": 1.0,
            "persistence_pressure": 0.2,
            "exploration_pressure": 0.3,
            "arbitration_adjustment": 0.0,
            "raw_control_residual": -0.1,
            "control_residual": -0.1,
            "staleness_signal": 0.5,
            "stay_tendency": -0.1,
        }
        for trial_index in range(12)
    ]

    rows = _summarize_reversal_windows(records, condition="full_control", seed=42)

    early = next(row for row in rows if row["window"] == "01-05")
    late = next(row for row in rows if row["window"] == "06-10")
    assert early["trial_count"] == 5
    assert early["correct_mean"] == 0.6
    assert late["trial_count"] == 5
    assert late["correct_mean"] == 1.0
    assert early["persistence_pressure_mean"] == 0.2
    assert early["exploration_pressure_mean"] == 0.3


def test_prl_diagnostic_dry_run_lists_checkpoint_rerolls(tmp_path: Path, capsys) -> None:
    args = PRLArbitrationDiagnosticArgs(
        source_root=tmp_path / "source",
        output_root=tmp_path / "output",
        conditions=("exploration_only",),
        seeds=(42,),
        dry_run=True,
    )

    args.run()

    output = capsys.readouterr().out
    assert "exploration_only_seed42" in output
    assert "control_diagnostics.ndjson" in output
