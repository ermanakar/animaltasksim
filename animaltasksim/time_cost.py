"""Utilities for modelling average-reward-based time costs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AverageRewardTimeCost:
    """Tracks an average reward rate and emits step-wise time penalties.

    The controller implements a simple running exponential average of observed
    reward rates (reward per second). Each environment step pays a penalty
    proportional to the current reward rate, encouraging agents to treat time as
    an explicit opportunity cost.
    """

    step_seconds: float
    alpha: float = 0.05
    scale: float = 1.0
    initial_rate: float = 1.0
    _avg_reward_rate: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.step_seconds <= 0.0:
            raise ValueError("step_seconds must be positive")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must fall within (0, 1]")
        if self.scale < 0.0:
            raise ValueError("scale must be non-negative")
        self._avg_reward_rate = max(self.initial_rate, 0.0)

    @property
    def avg_reward_rate(self) -> float:
        """Return the current running estimate of reward rate."""

        return self._avg_reward_rate

    def step_penalty(self) -> float:
        """Return the penalty to charge for the next step."""

        effective_rate = max(self._avg_reward_rate, 0.0)
        return self.scale * effective_rate * self.step_seconds

    def update(self, reward: float, elapsed_steps: int) -> None:
        """Update the running reward rate estimate.

        Args:
            reward: Total reward accrued over the elapsed interval (can be
                negative if the agent performed poorly or incurred penalties).
            elapsed_steps: Number of environment steps taken during the
                interval. The controller converts this to elapsed time using
                ``step_seconds``.
        """

        elapsed_steps = max(int(elapsed_steps), 1)
        elapsed_time = elapsed_steps * self.step_seconds
        reward_rate = reward / elapsed_time
        self._avg_reward_rate = (1.0 - self.alpha) * self._avg_reward_rate + self.alpha * reward_rate

    def reset(self, *, rate: float | None = None) -> None:
        """Reset the running reward rate to ``rate`` (or the initial value)."""

        if rate is None:
            self._avg_reward_rate = max(self.initial_rate, 0.0)
        else:
            self._avg_reward_rate = max(rate, 0.0)
