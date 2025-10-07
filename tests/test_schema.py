from __future__ import annotations

import json

import pytest

from eval.schema_validator import SchemaValidationError, validate_file, validate_line


def _write_ndjson(path, records):
    path.write_text("\n".join(json.dumps(rec) for rec in records) + "\n", encoding="utf-8")


def test_validate_line_accepts_valid_payload():
    payload = json.dumps(
        {
            "task": "ibl_2afc",
            "session_id": "session-1",
            "trial_index": 0,
            "stimulus": {"contrast": -0.25, "side": "left"},
            "block_prior": {"p_right": 0.2},
            "action": 0,
            "correct": True,
            "reward": 1.0,
            "rt_ms": 540.0,
            "phase_times": {"stim_ms": 300, "resp_ms": 700},
            "prev": None,
            "seed": 1234,
            "agent": {"name": "sticky_q", "version": "0.1.0"},
        }
    )

    record = validate_line(payload)
    assert record.task == "ibl_2afc"
    assert record.agent.name == "sticky_q"


def test_validate_line_rejects_missing_fields():
    payload = json.dumps(
        {
            "task": "ibl_2afc",
            "session_id": "session-1",
            "trial_index": 0,
            "stimulus": {"contrast": -0.25},
            "block_prior": None,
            "action": 0,
            "correct": True,
            # reward missing
            "rt_ms": 540.0,
            "phase_times": None,
            "prev": None,
            "seed": 1234,
            "agent": {"name": "sticky_q", "version": "0.1.0"},
        }
    )

    with pytest.raises(SchemaValidationError):
        validate_line(payload)


def test_validate_file_reports_all_errors(tmp_path):
    good_record = {
        "task": "ibl_2afc",
        "session_id": "session-2",
        "trial_index": 1,
        "stimulus": {"contrast": 0.5, "side": "right"},
        "block_prior": {"p_right": 0.8},
        "action": 1,
        "correct": True,
        "reward": 1.0,
        "rt_ms": 420.0,
        "phase_times": {"stim_ms": 300, "resp_ms": 650},
        "prev": {"action": 0, "reward": 1.0},
        "seed": 1234,
        "agent": {"name": "sticky_q", "version": "0.1.0"},
    }
    bad_record = {
        "task": "ibl_2afc",
        "session_id": "session-2",
        "trial_index": 2,
        "stimulus": {"contrast": 0.0},
        "block_prior": None,
        "action": 0,
        "correct": True,
        "reward": 0.0,
        "rt_ms": None,
        "phase_times": None,
        "prev": None,
        "seed": 1234,
        "agent": {"name": "", "version": "0.1.0"},
    }

    file_path = tmp_path / "sample.ndjson"
    _write_ndjson(file_path, [good_record, bad_record])

    result = validate_file(file_path, raise_on_error=False)
    assert result.total == 2
    assert len(result.errors) == 1
    assert "agent" in result.errors[0]

    with pytest.raises(SchemaValidationError):
        validate_file(file_path)


def test_validate_file_passes_without_errors(tmp_path):
    records = [
        {
            "task": "ibl_2afc",
            "session_id": "session-3",
            "trial_index": 0,
            "stimulus": {"contrast": -0.125, "side": "left"},
            "block_prior": {"p_right": 0.2},
            "action": 0,
            "correct": True,
            "reward": 1.0,
            "rt_ms": 560.0,
            "phase_times": {"stim_ms": 320, "resp_ms": 640},
            "prev": None,
            "seed": 2222,
            "agent": {"name": "sticky_q", "version": "0.1.0"},
        },
        {
            "task": "ibl_2afc",
            "session_id": "session-3",
            "trial_index": 1,
            "stimulus": {"contrast": 0.5, "side": "right"},
            "block_prior": {"p_right": 0.2},
            "action": 1,
            "correct": False,
            "reward": 0.0,
            "rt_ms": 480.0,
            "phase_times": {"stim_ms": 300, "resp_ms": 600},
            "prev": {"action": 0, "reward": 1.0},
            "seed": 2222,
            "agent": {"name": "sticky_q", "version": "0.1.0"},
        },
    ]

    file_path = tmp_path / "valid.ndjson"
    _write_ndjson(file_path, records)

    result = validate_file(file_path)
    assert result.ok
    assert result.total == 2
