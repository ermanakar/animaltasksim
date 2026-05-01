"""Train the adaptive control agent."""
from __future__ import annotations

import tyro

from agents.adaptive_control_agent import AdaptiveControlConfig, train_adaptive_control


def main(args: AdaptiveControlConfig) -> None:
    """Execute adaptive-control training."""
    result = train_adaptive_control(args)
    print("\n" + "=" * 80)
    print("ADAPTIVE CONTROL TRAINING COMPLETE")
    print("=" * 80)
    for key, path in result["paths"].items():
        print(f"{key}: {path}")
    print("=" * 80 + "\n")


def _parse_args() -> AdaptiveControlConfig:
    return tyro.cli(AdaptiveControlConfig)


if __name__ == "__main__":
    main(_parse_args())
