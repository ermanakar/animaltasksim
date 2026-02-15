"""Tests for v0.2 PRL/DMS schema extensions.

Ensures new optional fields are accepted when present, existing logs
without them still validate, and type constraints are enforced.
"""

from __future__ import annotations

import json

import pytest

from eval.schema_validator import SchemaValidationError, TrialRecord, validate_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_record(**overrides: object) -> dict[str, object]:
    """Return a minimal valid TrialRecord dict, with optional overrides."""
    rec: dict[str, object] = {
        "task": "ibl_2afc",
        "session_id": "session-1",
        "trial_index": 0,
        "stimulus": {"contrast": 0.25, "side": "left"},
        "block_prior": {"p_right": 0.2},
        "action": 0,
        "correct": True,
        "reward": 1.0,
        "rt_ms": 540.0,
        "phase_times": {"stim_ms": 300, "resp_ms": 700},
        "prev": None,
        "seed": 42,
        "agent": {"name": "sticky_q", "version": "0.1.0"},
    }
    rec.update(overrides)
    return rec


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Existing logs without v0.2 fields must still validate."""

    def test_existing_ibl_record_validates(self) -> None:
        record = validate_line(json.dumps(_base_record()))
        assert record.reversal is None
        assert record.block_index is None
        assert record.contingency is None
        assert record.sample_stimulus is None
        assert record.delay_ms is None
        assert record.match is None

    def test_existing_rdm_record_validates(self) -> None:
        rec = _base_record(
            task="rdm_macaque",
            stimulus={"coherence": 0.512, "direction": "right"},
            block_prior=None,
        )
        record = validate_line(json.dumps(rec))
        assert record.task == "rdm_macaque"

    def test_extra_field_still_forbidden(self) -> None:
        rec = _base_record(unknown_field="bad")
        with pytest.raises(SchemaValidationError):
            validate_line(json.dumps(rec))


# ---------------------------------------------------------------------------
# PRL extensions
# ---------------------------------------------------------------------------

class TestPRLFields:
    """Probabilistic Reversal Learning schema extensions."""

    def test_reversal_flag_accepted(self) -> None:
        rec = _base_record(task="prl", reversal=True, block_index=3)
        record = validate_line(json.dumps(rec))
        assert record.reversal is True
        assert record.block_index == 3

    def test_reversal_false(self) -> None:
        rec = _base_record(task="prl", reversal=False, block_index=0)
        record = validate_line(json.dumps(rec))
        assert record.reversal is False

    def test_contingency_dict(self) -> None:
        rec = _base_record(
            task="prl",
            reversal=False,
            block_index=1,
            contingency={"left": 0.8, "right": 0.2},
        )
        record = validate_line(json.dumps(rec))
        assert record.contingency == {"left": 0.8, "right": 0.2}

    def test_contingency_none(self) -> None:
        rec = _base_record(task="prl", reversal=True, contingency=None)
        record = validate_line(json.dumps(rec))
        assert record.contingency is None

    def test_reversal_wrong_type_rejected(self) -> None:
        rec = _base_record(reversal=[1, 2])  # list cannot coerce to bool
        with pytest.raises(SchemaValidationError):
            validate_line(json.dumps(rec))

    def test_block_index_wrong_type_rejected(self) -> None:
        rec = _base_record(block_index="two")
        with pytest.raises(SchemaValidationError):
            validate_line(json.dumps(rec))

    def test_partial_prl_fields_ok(self) -> None:
        """Only reversal without block_index is valid."""
        rec = _base_record(reversal=True)
        record = validate_line(json.dumps(rec))
        assert record.reversal is True
        assert record.block_index is None


# ---------------------------------------------------------------------------
# DMS extensions
# ---------------------------------------------------------------------------

class TestDMSFields:
    """Delayed Match-to-Sample schema extensions."""

    def test_dms_full_record(self) -> None:
        rec = _base_record(
            task="dms",
            sample_stimulus={"shape": "circle", "color": "red"},
            delay_ms=1500.0,
            match=True,
        )
        record = validate_line(json.dumps(rec))
        assert record.sample_stimulus == {"shape": "circle", "color": "red"}
        assert record.delay_ms == 1500.0
        assert record.match is True

    def test_dms_non_match_trial(self) -> None:
        rec = _base_record(
            task="dms",
            sample_stimulus={"shape": "square"},
            delay_ms=2000.0,
            match=False,
        )
        record = validate_line(json.dumps(rec))
        assert record.match is False

    def test_delay_ms_none(self) -> None:
        rec = _base_record(sample_stimulus={"shape": "circle"}, delay_ms=None)
        record = validate_line(json.dumps(rec))
        assert record.delay_ms is None

    def test_sample_stimulus_none(self) -> None:
        rec = _base_record(sample_stimulus=None, delay_ms=500.0)
        record = validate_line(json.dumps(rec))
        assert record.sample_stimulus is None

    def test_delay_ms_wrong_type_rejected(self) -> None:
        rec = _base_record(delay_ms="long")
        with pytest.raises(SchemaValidationError):
            validate_line(json.dumps(rec))

    def test_match_wrong_type_rejected(self) -> None:
        rec = _base_record(match={"value": 1})  # dict cannot coerce to bool
        with pytest.raises(SchemaValidationError):
            validate_line(json.dumps(rec))

    def test_partial_dms_fields_ok(self) -> None:
        """Only sample_stimulus without delay is valid."""
        rec = _base_record(sample_stimulus={"shape": "triangle"})
        record = validate_line(json.dumps(rec))
        assert record.sample_stimulus == {"shape": "triangle"}
        assert record.delay_ms is None
        assert record.match is None


# ---------------------------------------------------------------------------
# Mixed task types
# ---------------------------------------------------------------------------

class TestMixedTasks:
    """Cross-task records coexist in the same schema."""

    def test_prl_and_dms_fields_can_coexist(self) -> None:
        """Though unusual, having both PRL and DMS fields is schema-valid."""
        rec = _base_record(
            reversal=False,
            block_index=2,
            sample_stimulus={"shape": "star"},
            delay_ms=1000.0,
            match=True,
        )
        record = validate_line(json.dumps(rec))
        assert record.reversal is False
        assert record.sample_stimulus == {"shape": "star"}

    def test_existing_ibl_unaffected_by_new_fields(self) -> None:
        """An IBL record still validates and has None for all v0.2 fields."""
        rec = _base_record()
        record = TrialRecord.model_validate(rec)
        v02_fields = ["reversal", "block_index", "contingency",
                       "sample_stimulus", "delay_ms", "match"]
        for field in v02_fields:
            assert getattr(record, field) is None, f"{field} should be None"
