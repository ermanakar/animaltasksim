"""Runtime validation for the AnimalTaskSim `.ndjson` trial log schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class AgentMetadata(BaseModel):
    """Metadata describing the agent that generated a trial."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str

    @field_validator("name", "version")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("must be a non-empty string")
        return value


class TrialRecord(BaseModel):
    """Schema for a single trial entry in the unified `.ndjson` log."""

    model_config = ConfigDict(extra="forbid")

    task: str
    session_id: str
    trial_index: int
    stimulus: dict[str, object]
    block_prior: dict[str, object] | None
    action: int | str
    correct: bool
    reward: float
    rt_ms: float | None
    phase_times: dict[str, object] | None
    prev: dict[str, object] | None
    seed: int
    agent: AgentMetadata

    @field_validator("session_id")
    @classmethod
    def _session_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("session_id must be non-empty")
        return value


@dataclass(slots=True)
class SchemaValidationResult:
    """Summary of schema validation for one or more records."""

    total: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


class SchemaValidationError(RuntimeError):
    """Raised when schema validation fails and `raise_on_error` is set."""


def _iter_lines(source: Iterable[str]) -> Iterator[tuple[int, str]]:
    for index, raw in enumerate(source, start=1):
        line = raw.strip()
        if not line:
            continue
        yield index, line


def validate_line(payload: str) -> TrialRecord:
    """Validate a single JSON payload against the schema."""

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - sanity check path
        raise SchemaValidationError(f"Invalid JSON payload: {exc}") from exc

    try:
        return TrialRecord.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc


def validate_file(path: str | Path, *, raise_on_error: bool = True) -> SchemaValidationResult:
    """Validate an `.ndjson` log file.

    Args:
        path: Path to the log file.
        raise_on_error: Whether to raise `SchemaValidationError` on failure.

    Returns:
        A `SchemaValidationResult` containing the number of validated records and encountered errors.
    """

    file_path = Path(path)
    errors: list[str] = []
    total = 0

    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, payload in _iter_lines(handle):
            total += 1
            try:
                validate_line(payload)
            except SchemaValidationError as exc:
                errors.append(f"line {line_no}: {exc}")

    if errors and raise_on_error:
        joined = "\n".join(errors)
        raise SchemaValidationError(f"Schema validation failed for {file_path}:\n{joined}")

    return SchemaValidationResult(total=total, errors=errors)


__all__ = [
    "AgentMetadata",
    "SchemaValidationError",
    "SchemaValidationResult",
    "TrialRecord",
    "validate_file",
    "validate_line",
]
