"""Utilities for shaping rewards based on average reward rate over time."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AverageRewardTimeCost:
    """Simple controller that penalizes time based on running average reward rate.

    The controller keeps an exponential moving average (EMA) of reward-per-second.
    On every environment step, :meth:`step_penalty` returns a non-negative term that
    can be subtracted from the agent's reward to encourage faster trial completion.

    After each trial, :meth:`update` should be called with the total reward collected
    over that trial together with the number of environment steps. This refreshes the
    EMA so the penalty tracks the current average return rate.
    """

    step_seconds: float
    alpha: float
    scale: float
    initial_rate: float
    _avg_rate: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must fall within (0, 1]")
        if self.scale < 0.0:
            raise ValueError("scale must be non-negative")
        if self.initial_rate < 0.0:
            raise ValueError("initial_rate must be non-negative")
        self._avg_rate = float(self.initial_rate)

    def reset(self) -> None:
        """Reset the moving-average reward rate back to ``initial_rate``."""

        self._avg_rate = float(self.initial_rate)

    def step_penalty(self) -> float:
        """Return the time penalty to apply for the current step.

        The penalty is proportional to both the estimated reward rate and the time
        spent on a single environment step. It is always non-negative.
        """

        if self.scale == 0.0:
            return 0.0
        return float(self.scale * max(self._avg_rate, 0.0) * self.step_seconds)

    def update(self, total_reward: float, steps: int) -> None:
        """Refresh the average reward rate after a completed trial.

        Parameters
        ----------
        total_reward:
            Sum of rewards obtained during the trial.
        steps:
            Number of environment steps taken during the trial. Must be positive.
        """

        if steps <= 0:
            return
        duration = steps * self.step_seconds
        if duration <= 0:
            return
        reward_rate = float(total_reward) / duration
        self._avg_rate = (1.0 - self.alpha) * self._avg_rate + self.alpha * max(reward_rate, 0.0)


__all__ = ["AverageRewardTimeCost"]
