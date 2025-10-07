"""Structured logging utilities for AnimalTaskSim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, MutableMapping

from eval.schema_validator import TrialRecord


class NDJSONTrialLogger:
    """Append-only `.ndjson` logger that validates each record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, record: Mapping[str, object]) -> None:
        payload: MutableMapping[str, object] = dict(record)
        TrialRecord.model_validate(payload)
        self._handle.write(json.dumps(payload, separators=(",", ":")))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "NDJSONTrialLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["NDJSONTrialLogger"]
