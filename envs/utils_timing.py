"""Timing helpers shared by task environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(slots=True)
class PhaseTiming:
    """Represents the timing characteristics of an environment phase."""

    name: str
    duration_steps: int

    def __post_init__(self) -> None:
        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")


def total_steps(schedule: Sequence[PhaseTiming]) -> int:
    """Return the total number of steps across the schedule."""

    return sum(phase.duration_steps for phase in schedule)


def ensure_phase_names(schedule: Iterable[PhaseTiming]) -> list[str]:
    """Return the ordered phase names, ensuring uniqueness."""

    names: list[str] = []
    for phase in schedule:
        if phase.name in names:
            raise ValueError(f"duplicate phase name: {phase.name}")
        names.append(phase.name)
    return names


__all__ = ["PhaseTiming", "ensure_phase_names", "total_steps"]
