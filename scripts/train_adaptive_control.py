"""Train the adaptive control agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tyro

from agents.adaptive_control_agent import AdaptiveControlConfig, train_adaptive_control


TrainControlProfile = Literal[
    "from_flags",
    "custom",
    "no_control",
    "persistence_only",
    "exploration_only",
    "full_control",
]


@dataclass(slots=True)
class TrainAdaptiveControlArgs(AdaptiveControlConfig):
    """Command-line arguments for adaptive-control training.

    Args:
        control_profile: Named control profile. Leave as `from_flags` to use
            the explicit boolean switches; the default switches resolve to the
            recommended `persistence_only` profile. Use `full_control` only as
            a comparison condition because exploration is experimental and not
            independently validated.
        control_state_enabled: Enables the adaptive-control fast state. Disable
            with persistence and exploration for the clean no-control lesion.
        persistence_enabled: Enables the validated persistence/retry controller.
        exploration_enabled: Enables the experimental exploration controller.
            Defaults to off; use `--control-profile full_control`,
            `--control-profile exploration_only`, or `--exploration-enabled`
            only for explicitly labeled comparison runs.
    """

    control_profile: TrainControlProfile = "from_flags"

    def resolve_control_profile(self) -> TrainAdaptiveControlArgs:
        """Apply an explicit profile and label the resolved condition."""
        if self.control_profile not in {"from_flags", "custom"}:
            self.apply_control_profile(self.control_profile)
        self.control_profile = self.active_control_profile
        return self


def main(args: TrainAdaptiveControlArgs) -> None:
    """Execute adaptive-control training."""
    config = args.resolve_control_profile()
    result = train_adaptive_control(config)
    print("\n" + "=" * 80)
    print("ADAPTIVE CONTROL TRAINING COMPLETE")
    print(f"control_profile: {config.control_profile}")
    print(f"exploration_enabled: {config.exploration_enabled}")
    print("=" * 80)
    for key, path in result["paths"].items():
        print(f"{key}: {path}")
    print("=" * 80 + "\n")


def _parse_args() -> TrainAdaptiveControlArgs:
    return tyro.cli(TrainAdaptiveControlArgs)


if __name__ == "__main__":
    main(_parse_args())
