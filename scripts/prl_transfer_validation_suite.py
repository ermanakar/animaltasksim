#!/usr/bin/env python3
"""Run matched adaptive-control transfer conditions in the PRL environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from scripts.adaptive_control_validation_suite import ValidationCondition, ValidationSuiteArgs


@dataclass(slots=True)
class PRLTransferSuiteArgs(ValidationSuiteArgs):
    """Matched PRL transfer suite including the May arbitration candidate."""

    run_root: Path = Path("runs/prl_transfer_validation_suite")
    task: Literal["prl"] = "prl"
    episodes: int = 4
    trials_per_episode: int = 400
    epochs: int = 1
    max_sessions: int = 5
    max_trials_per_session: int = 128

    def _conditions(self) -> list[ValidationCondition]:
        """Return claim baselines plus the best current full-control transfer candidate."""
        return [
            ValidationCondition(
                label="true_no_control",
                description="Clean lesion: disables all adaptive-control fast state and overlays.",
                control_profile="no_control",
            ),
            ValidationCondition(
                label="exploration_only",
                description="Experimental exploration lesion; persistence disabled.",
                control_profile="exploration_only",
            ),
            ValidationCondition(
                label="persistence_only",
                description="Validated retry profile; exploration disabled.",
                control_profile="persistence_only",
            ),
            ValidationCondition(
                label="full_control_default",
                description="Default full-control comparison: persistence=1.6, exploration=0.8.",
                control_profile="full_control",
                persistence_bias_scale=1.6,
                exploration_bias_scale=0.8,
            ),
            ValidationCondition(
                label="full_control_persist_half",
                description="May 7 transfer candidate: persistence=0.8, exploration=0.8.",
                control_profile="full_control",
                persistence_bias_scale=0.8,
                exploration_bias_scale=0.8,
            ),
        ]


def main() -> None:
    """CLI entry point."""
    tyro.cli(PRLTransferSuiteArgs).run()


__all__ = ["PRLTransferSuiteArgs", "main"]


if __name__ == "__main__":
    main()
