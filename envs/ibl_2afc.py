"""IBL-style mouse 2AFC environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding

from animaltasksim.logging import NDJSONTrialLogger
from envs.utils_timing import PhaseTiming, ensure_phase_names

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_NO_OP = 2
ACTION_NAMES = {
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
    ACTION_NO_OP: "no-op",
}

DEFAULT_PHASE_SCHEDULE = (
    PhaseTiming("iti", 10),
    PhaseTiming("stimulus", 10),
    PhaseTiming("response", 30),
    PhaseTiming("outcome", 10),
)


def _validate_contrasts(contrasts: Iterable[float]) -> list[float]:
    values = [float(v) for v in contrasts]
    if not values:
        raise ValueError("contrast_set must contain at least one value")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError("contrast magnitudes must be within [0, 1]")
    return values


@dataclass(slots=True)
class BlockConfig:
    """Configuration for a bias block."""

    p_right: float
    length: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_right <= 1.0):
            raise ValueError("p_right must be within [0, 1]")
        if self.length <= 0:
            raise ValueError("length must be positive")


@dataclass(slots=True)
class AgentMetadata:
    """Metadata identifying the acting agent."""

    name: str = "unknown"
    version: str = "0.0"

    def to_dict(self) -> dict[str, str]:
        if not self.name:
            raise ValueError("agent name must be non-empty")
        if not self.version:
            raise ValueError("agent version must be non-empty")
        return {"name": self.name, "version": self.version}


@dataclass(slots=True)
class IBL2AFCConfig:
    """Configuration bundle for the IBL 2AFC environment."""

    trials_per_episode: int = 400
    contrast_set: tuple[float, ...] = (
        0.0,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
    )
    block_sequence: tuple[BlockConfig, ...] = (
        BlockConfig(p_right=0.5, length=80),
        BlockConfig(p_right=0.8, length=40),
        BlockConfig(p_right=0.2, length=40),
    )
    phase_schedule: tuple[PhaseTiming, ...] = DEFAULT_PHASE_SCHEDULE
    step_ms: int = 10
    include_phase_onehot: bool = False
    include_timing: bool = False
    expose_prior: bool = False
    include_history: bool = False
    min_response_latency_steps: int = 0  # Optional non-decision latency before responses are captured
    log_path: Path | None = None
    agent: AgentMetadata = field(default_factory=AgentMetadata)
    seed: int | None = None

    def __post_init__(self) -> None:
        ensure_phase_names(self.phase_schedule)
        if self.trials_per_episode <= 0:
            raise ValueError("trials_per_episode must be positive")
        if self.step_ms <= 0:
            raise ValueError("step_ms must be positive")
        _validate_contrasts(self.contrast_set)
        if not self.block_sequence:
            raise ValueError("block_sequence must contain at least one entry")
        if self.min_response_latency_steps < 0:
            raise ValueError("min_response_latency_steps must be non-negative")
        if self.min_response_latency_steps > 0:
            response_phase = next((p for p in self.phase_schedule if p.name == "response"), None)
            if response_phase is None:
                raise ValueError("phase_schedule must include a 'response' phase")
            if self.min_response_latency_steps >= response_phase.duration_steps:
                raise ValueError("min_response_latency_steps must be smaller than response phase duration")


class IBL2AFCEnv(Env):
    """Gymnasium environment capturing the IBL 2AFC task."""

    metadata = {"render_modes": []}

    def __init__(self, config: IBL2AFCConfig | None = None) -> None:
        super().__init__()
        self.config = config or IBL2AFCConfig()
        self._phase_schedule = self.config.phase_schedule
        self._phase_names = ensure_phase_names(self._phase_schedule)
        self._contrast_magnitudes = _validate_contrasts(self.config.contrast_set)
        self.action_space = spaces.Discrete(3)
        self.observation_space = self._build_observation_space()

        self._logger: NDJSONTrialLogger | None = None
        if self.config.log_path is not None:
            self._logger = NDJSONTrialLogger(self.config.log_path)

        self._session_id: str = ""
        self._rng: np.random.Generator | None = None
        self._trial_index: int = 0
        self._block_index: int = 0
        self._trials_into_block: int = 0
        self._phase_index: int = 0
        self._phase_step: int = 0
        self._phase_step_counts: dict[str, int] = {}
        self._response_action: str = ACTION_NAMES[ACTION_NO_OP]
        self._response_captured: bool = False
        self._correct: bool = False
        self._trial_reward: float = 0.0
        self._rt_steps: int | None = None
        self._stimulus: dict[str, object] = {}
        self._block_prior: float = 0.5
        self._prev_action: str | None = None
        self._prev_reward: float | None = None
        self._prev_correct: bool | None = None
        self._terminated: bool = False
        self._seed: int | None = None

    def _build_observation_space(self) -> spaces.Dict:
        space_dict: dict[str, spaces.Space] = {
            "contrast": spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
        }
        if self.config.include_phase_onehot:
            space_dict["phase_onehot"] = spaces.Box(
                low=0.0, high=1.0, shape=(len(self._phase_schedule),), dtype=np.float32
            )
        if self.config.include_timing:
            space_dict["t_norm"] = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        if self.config.expose_prior:
            space_dict["block_prior"] = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        if self.config.include_history:
            space_dict["prev_action"] = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
            space_dict["prev_reward"] = spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
            space_dict["prev_correct"] = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        return spaces.Dict(space_dict)

    @property
    def _current_phase(self) -> PhaseTiming:
        return self._phase_schedule[self._phase_index]

    def _current_phase_name(self) -> str:
        return self._current_phase.name

    def _build_observation(self) -> dict[str, np.ndarray | np.float32]:
        phase_name = self._current_phase_name()
        contrast = 0.0
        if phase_name in {"stimulus", "response", "outcome"}:
            contrast = float(self._stimulus.get("contrast", 0.0))
        obs: dict[str, np.ndarray | np.float32] = {"contrast": np.array(contrast, dtype=np.float32)}
        if self.config.include_phase_onehot:
            onehot = np.zeros(len(self._phase_schedule), dtype=np.float32)
            onehot[self._phase_index] = 1.0
            obs["phase_onehot"] = onehot
        if self.config.include_timing:
            duration = self._current_phase.duration_steps
            obs["t_norm"] = np.array(self._phase_step / max(duration - 1, 1), dtype=np.float32)
        if self.config.expose_prior:
            obs["block_prior"] = np.array(self._block_prior, dtype=np.float32)
        if self.config.include_history:
            prev_action_onehot = np.zeros(3, dtype=np.float32)
            if self._prev_action is not None:
                action_map = {
                    ACTION_NAMES[a]: a for a in [ACTION_LEFT, ACTION_RIGHT, ACTION_NO_OP]
                }
                if self._prev_action in action_map:
                    prev_action_onehot[action_map[self._prev_action]] = 1.0
            obs["prev_action"] = prev_action_onehot
            prev_reward = self._prev_reward if self._prev_reward is not None else 0.0
            obs["prev_reward"] = np.array(prev_reward, dtype=np.float32)
            prev_correct = self._prev_correct if self._prev_correct is not None else 0.0
            obs["prev_correct"] = np.array(prev_correct, dtype=np.float32)
        return obs

    def _default_info(self) -> dict[str, object]:
        return {
            "session_id": self._session_id,
            "trial_index": self._trial_index,
            "phase": self._current_phase_name(),
            "phase_step": self._phase_step,
            "block_index": self._block_index,
            "block_prior": self._block_prior,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        request_seed = seed if seed is not None else self.config.seed
        self._rng, actual_seed = seeding.np_random(request_seed)
        self._seed = int(actual_seed)
        options = options or {}
        self._session_id = options.get("session_id", str(uuid4()))

        self._trial_index = 0
        self._block_index = 0
        self._trials_into_block = 0
        self._phase_index = 0
        self._phase_step = 0
        self._phase_step_counts = {name: 0 for name in self._phase_names}
        self._response_action = ACTION_NAMES[ACTION_NO_OP]
        self._response_captured = False
        self._correct = False
        self._trial_reward = 0.0
        self._rt_steps = None
        self._prev_action = None
        self._prev_reward = None
        self._prev_correct = None
        self._terminated = False

        self._start_new_trial()
        return self._build_observation(), self._default_info()

    def _start_new_trial(self) -> None:
        block = self.config.block_sequence[self._block_index]
        self._block_prior = block.p_right
        magnitude = float(self._rng.choice(self._contrast_magnitudes)) if self._rng else 0.0
        side = "right" if self._rng.random() < block.p_right else "left"
        sign = 1.0 if side == "right" else -1.0
        contrast_value = magnitude * sign
        if magnitude == 0.0:
            contrast_value = 0.0
            side = "none"
        self._stimulus = {"contrast": contrast_value, "side": side}
        self._phase_index = 0
        self._phase_step = 0
        self._phase_step_counts = {name: 0 for name in self._phase_names}
        self._response_action = ACTION_NAMES[ACTION_NO_OP]
        self._response_captured = False
        self._correct = False
        self._trial_reward = 0.0
        self._rt_steps = None

    def _advance_block(self) -> None:
        block = self.config.block_sequence[self._block_index]
        self._trials_into_block += 1
        if self._trials_into_block >= block.length:
            self._block_index = (self._block_index + 1) % len(self.config.block_sequence)
            self._trials_into_block = 0

    def _process_response(self, action: int) -> None:
        if action not in ACTION_NAMES:
            raise ValueError(f"invalid action {action}")
        if self._response_captured:
            return
        if action == ACTION_NO_OP:
            return
        if self.config.min_response_latency_steps > 0:
            response_steps = self._phase_step_counts.get("response", 0)
            if response_steps < self.config.min_response_latency_steps:
                # Enforce a minimum non-decision time before registering the choice.
                return

        self._response_action = ACTION_NAMES[action]
        self._response_captured = True
        self._rt_steps = self._phase_step + 1

        contrast = float(self._stimulus.get("contrast", 0.0))
        expected = ACTION_RIGHT if contrast > 0 else ACTION_LEFT
        if contrast == 0.0:
            # On zero-contrast trials, reward choice towards the high-probability side
            expected = ACTION_RIGHT if self._block_prior > 0.5 else ACTION_LEFT
        self._correct = action == expected
        self._trial_reward = 1.0 if self._correct else -0.1

        # Jump to outcome on the next step.
        self._phase_step = self._current_phase.duration_steps - 1

    def _finalize_without_response(self) -> None:
        if self._response_captured:
            return
        self._response_action = ACTION_NAMES[ACTION_NO_OP]
        self._response_captured = True
        self._correct = False
        self._trial_reward = 0.0
        self._rt_steps = None

    def _log_trial(self) -> None:
        if self._logger is None:
            return
        phase_times = {
            f"{name}_ms": self.config.step_ms * steps
            for name, steps in self._phase_step_counts.items()
        }
        record = {
            "task": "ibl_2afc",
            "session_id": self._session_id,
            "trial_index": self._trial_index,
            "stimulus": self._stimulus,
            "block_prior": {"p_right": self._block_prior},
            "action": self._response_action,
            "correct": self._correct,
            "reward": float(self._trial_reward),
            "rt_ms": None if self._rt_steps is None else float(self._rt_steps * self.config.step_ms),
            "phase_times": phase_times,
            "prev": None
            if self._prev_action is None
            else {"action": self._prev_action, "reward": self._prev_reward, "correct": self._prev_correct},
            "seed": int(self._seed or 0),
            "agent": self.config.agent.to_dict(),
        }
        self._logger.log(record)
        self._prev_action = self._response_action
        self._prev_reward = float(self._trial_reward)
        self._prev_correct = self._correct

    def step(self, action: int):  # type: ignore[override]
        if self._terminated:
            raise RuntimeError("Episode already terminated; call reset().")

        phase_name = self._current_phase_name()
        if phase_name == "response":
            self._process_response(int(action))
        else:
            # Actions outside response are ignored; treat as no-op.
            action = ACTION_NO_OP

        reward = 0.0
        if phase_name == "outcome" and self._phase_step == 0:
            reward = float(self._trial_reward)

        self._phase_step_counts[phase_name] += 1

        terminated = False
        truncated = False

        # Advance phase and detect trial completion.
        self._phase_step += 1
        trial_completed = False
        if self._phase_step >= self._current_phase.duration_steps:
            self._phase_step = 0
            self._phase_index += 1
            if self._phase_index >= len(self._phase_schedule):
                self._phase_index = len(self._phase_schedule) - 1
                self._phase_step = self._phase_schedule[-1].duration_steps - 1
                trial_completed = True

        if trial_completed:
            if not self._response_captured:
                self._finalize_without_response()
            self._log_trial()
            self._advance_block()
            terminated = (self._trial_index + 1) >= self.config.trials_per_episode
            if terminated:
                self._terminated = True
                observation = self._terminal_observation()
                info = {
                    "session_id": self._session_id,
                    "trial_index": self._trial_index,
                    "phase": "terminal",
                    "phase_step": 0,
                    "block_index": self._block_index,
                    "block_prior": self._block_prior,
                }
            else:
                self._trial_index += 1
                self._start_new_trial()
                observation = self._build_observation()
                info = self._default_info()
        else:
            observation = self._build_observation()
            info = self._default_info()

        return observation, reward, terminated, truncated, info

    def _terminal_observation(self) -> dict[str, np.ndarray | np.float32]:
        obs: dict[str, np.ndarray | np.float32] = {"contrast": np.zeros((), dtype=np.float32)}
        if self.config.include_phase_onehot:
            obs["phase_onehot"] = np.zeros(len(self._phase_schedule), dtype=np.float32)
        if self.config.include_timing:
            obs["t_norm"] = np.zeros((), dtype=np.float32)
        if self.config.expose_prior:
            obs["block_prior"] = np.array(self._block_prior, dtype=np.float32)
        if self.config.include_history:
            obs["prev_action"] = np.zeros(3, dtype=np.float32)
            obs["prev_reward"] = np.zeros((), dtype=np.float32)
            obs["prev_correct"] = np.zeros((), dtype=np.float32)
        return obs

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()
        super().close()


__all__ = [
    "ACTION_LEFT",
    "ACTION_NO_OP",
    "ACTION_RIGHT",
    "ACTION_NAMES",
    "AgentMetadata",
    "BlockConfig",
    "IBL2AFCConfig",
    "IBL2AFCEnv",
]
