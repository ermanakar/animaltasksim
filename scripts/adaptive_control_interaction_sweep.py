#!/usr/bin/env python3
"""Run persistence/exploration interaction conditions for adaptive control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from scripts.adaptive_control_validation_suite import ValidationCondition, ValidationSuiteArgs


@dataclass(slots=True)
class InteractionSweepArgs(ValidationSuiteArgs):
    """Matched sweep over persistence/exploration scale interactions."""

    run_root: Path = Path("runs/adaptive_control_interaction_sweep")

    def _conditions(self) -> list[ValidationCondition]:
        """Return baseline lesions plus full-control interaction conditions."""
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
                description="Validated retry lesion; exploration disabled.",
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
                description="Full control with weaker persistence: persistence=0.8, exploration=0.8.",
                control_profile="full_control",
                persistence_bias_scale=0.8,
                exploration_bias_scale=0.8,
            ),
            ValidationCondition(
                label="full_control_persist_quarter",
                description="Full control with much weaker persistence: persistence=0.4, exploration=0.8.",
                control_profile="full_control",
                persistence_bias_scale=0.4,
                exploration_bias_scale=0.8,
            ),
            ValidationCondition(
                label="full_control_explore_strong",
                description="Full control with stronger exploration: persistence=1.6, exploration=1.2.",
                control_profile="full_control",
                persistence_bias_scale=1.6,
                exploration_bias_scale=1.2,
            ),
            ValidationCondition(
                label="full_control_explore_double",
                description="Full control with double exploration: persistence=1.6, exploration=1.6.",
                control_profile="full_control",
                persistence_bias_scale=1.6,
                exploration_bias_scale=1.6,
            ),
            ValidationCondition(
                label="full_control_balanced",
                description="Full control with weaker persistence and stronger exploration: persistence=0.8, exploration=1.2.",
                control_profile="full_control",
                persistence_bias_scale=0.8,
                exploration_bias_scale=1.2,
            ),
            ValidationCondition(
                label="full_control_explore_dominant",
                description="Full control with exploration-dominant scales: persistence=0.4, exploration=1.6.",
                control_profile="full_control",
                persistence_bias_scale=0.4,
                exploration_bias_scale=1.6,
            ),
        ]


def main() -> None:
    """CLI entry point."""
    tyro.cli(InteractionSweepArgs).run()


__all__ = ["InteractionSweepArgs", "main"]


if __name__ == "__main__":
    main()
