"""Configuration primitives shared across AnimalTaskSim modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProjectPaths:
    """Convenience container for common project directories.

    Attributes:
        root: Repository root directory.
        data: Directory containing reference datasets.
        runs: Default output directory for training/evaluation runs.
        logs: Directory for structured logs or debugging artefacts.
    """

    root: Path
    data: Path
    runs: Path
    logs: Path

    @classmethod
    def from_cwd(cls, cwd: str | Path | None = None, *, create: bool = True) -> "ProjectPaths":
        """Create a `ProjectPaths` instance relative to `cwd`.

        Args:
            cwd: Base directory. Defaults to the current working directory.
            create: Whether to create the derived directories if missing.

        Returns:
            A populated `ProjectPaths` instance with optional directory creation.
        """

        base = Path(cwd or Path.cwd()).resolve()
        data = base / "data"
        runs = base / "runs"
        logs = base / "logs"

        if create:
            for path in (data, runs, logs):
                path.mkdir(parents=True, exist_ok=True)

        return cls(root=base, data=data, runs=runs, logs=logs)


__all__ = ["ProjectPaths"]
