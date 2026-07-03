"""Delayed Match-to-Sample (DMS) short-term memory environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding

from animaltasksim.logging import NDJSONTrialLogger
from envs.ibl_2afc import ACTION_LEFT, ACTION_NAMES, ACTION_NO_OP, ACTION_RIGHT, AgentMetadata
from envs.utils_timing import PhaseTiming, ensure_phase_names

DEFAULT_DMS_PHASE_SCHEDULE = (
    PhaseTiming("iti", 10),
    PhaseTiming("sample", 15),
    PhaseTiming("delay", 50),
    PhaseTiming("test", 15),
    PhaseTiming("response", 30),
    PhaseTiming("outcome", 10),
)


@dataclass(slots=True)
class DMSConfig:
    """Configuration bundle for the DMS environment."""

    trials_per_episode: int = 400
    contrast_set: tuple[float, ...] = (
        0.25,
        1.0,
    )
    phase_schedule: tuple[PhaseTiming, ...] = DEFAULT_DMS_PHASE_SCHEDULE
    step_ms: int = 10
    include_phase_onehot: bool = False
    include_timing: bool = False
    expose_prior: bool = False
    include_history: bool = False
    min_response_latency_steps: int = 0
    log_path: Path | None = None
    agent: AgentMetadata = field(default_factory=AgentMetadata)
    seed: int | None = None

    def __post_init__(self) -> None:
        phase_names = ensure_phase_names(self.phase_schedule)
        if self.trials_per_episode <= 0:
            raise ValueError("trials_per_episode must be positive")
        if self.step_ms <= 0:
            raise ValueError("step_ms must be positive")
        if not self.contrast_set:
            raise ValueError("contrast_set must contain at least one value")
        if any(contrast < 0.0 or contrast > 1.0 for contrast in self.contrast_set):
            raise ValueError("contrast magnitudes must be within [0, 1]")
        if self.min_response_latency_steps < 0:
            raise ValueError("min_response_latency_steps must be non-negative")
        required_phases = {"iti", "sample", "delay", "test", "response", "outcome"}
        if not required_phases.issubset(phase_names):
            raise ValueError("phase_schedule must include iti, sample, delay, test, response, and outcome phases")
        response_phase = next(phase for phase in self.phase_schedule if phase.name == "response")
        if self.min_response_latency_steps >= response_phase.duration_steps:
            raise ValueError("min_response_latency_steps must be smaller than response phase duration")


class DelayedMatchToSampleEnv(Env):
    """Gymnasium environment capturing the Delayed Match-to-Sample task."""

    metadata = {"render_modes": []}

    def __init__(self, config: DMSConfig | None = None) -> None:
        super().__init__()
        self.config = config or DMSConfig()
        self._phase_schedule = self.config.phase_schedule
        self._phase_names = ensure_phase_names(self._phase_schedule)
        self.action_space = spaces.Discrete(3)
        self.observation_space = self._build_observation_space()

        self._logger: NDJSONTrialLogger | None = None
        if self.config.log_path is not None:
            self._logger = NDJSONTrialLogger(self.config.log_path)

        self._session_id: str = ""
        self._rng: np.random.Generator | None = None
        self._trial_index: int = 0
        self._phase_index: int = 0
        self._phase_step: int = 0
        self._phase_step_counts: dict[str, int] = {}
        self._response_action: str = ACTION_NAMES[ACTION_NO_OP]
        self._response_captured: bool = False
        self._correct: bool = False
        self._trial_reward: float = 0.0
        self._rt_steps: int | None = None

        self._sample_stimulus: dict[str, object] = {}
        self._test_stimulus: dict[str, object] = {}
        self._is_match: bool = False
        self._delay_ms: float = 0.0

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

        # Display sample visual contrast during "sample" phase
        if phase_name == "sample":
            contrast = float(self._sample_stimulus.get("contrast", 0.0))
        # Display test visual contrast during "test", "response", and "outcome" phases
        elif phase_name in {"test", "response", "outcome"}:
            contrast = float(self._test_stimulus.get("contrast", 0.0))

        obs: dict[str, np.ndarray | np.float32] = {"contrast": np.array(contrast, dtype=np.float32)}
        if self.config.include_phase_onehot:
            onehot = np.zeros(len(self._phase_schedule), dtype=np.float32)
            onehot[self._phase_index] = 1.0
            obs["phase_onehot"] = onehot
        if self.config.include_timing:
            duration = self._current_phase.duration_steps
            obs["t_norm"] = np.array(self._phase_step / max(duration - 1, 1), dtype=np.float32)
        if self.config.expose_prior:
            # Match baseline prior
            obs["block_prior"] = np.array(0.5, dtype=np.float32)
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
            "sample_stimulus": self._sample_stimulus,
            "delay_ms": self._delay_ms,
            "match": self._is_match,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        request_seed = seed if seed is not None else self.config.seed
        self._rng, actual_seed = seeding.np_random(request_seed)
        self._seed = int(actual_seed)
        options = options or {}
        self._session_id = options.get("session_id", str(uuid4()))

        self._trial_index = 0
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
        magnitude = float(self._rng.choice(self.config.contrast_set)) if self._rng else 1.0
        side = "right" if self._rng.random() < 0.5 else "left"
        sign = 1.0 if side == "right" else -1.0

        self._sample_stimulus = {"contrast": magnitude * sign, "side": side}

        # Decide if this trial is a Match (50% probability)
        self._is_match = self._rng.random() < 0.5 if self._rng else True

        if self._is_match:
            test_side = side
            test_sign = sign
        else:
            test_side = "left" if side == "right" else "right"
            test_sign = -sign

        # Preserve visual strength so match/non-match classification requires
        # memory of the sample identity rather than a contrast mismatch shortcut.
        self._test_stimulus = {"contrast": magnitude * test_sign, "side": test_side}

        # Calculate delay time in milliseconds
        delay_phase = next(p for p in self._phase_schedule if p.name == "delay")
        self._delay_ms = float(delay_phase.duration_steps * self.config.step_ms)

        self._phase_index = 0
        self._phase_step = 0
        self._phase_step_counts = {name: 0 for name in self._phase_names}
        self._response_action = ACTION_NAMES[ACTION_NO_OP]
        self._response_captured = False
        self._correct = False
        self._trial_reward = 0.0
        self._rt_steps = None

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
                return

        self._response_action = ACTION_NAMES[action]
        self._response_captured = True
        self._rt_steps = self._phase_step + 1

        # Action Left (0) = Non-Match; Action Right (1) = Match
        expected_action = ACTION_RIGHT if self._is_match else ACTION_LEFT
        self._correct = action == expected_action
        self._trial_reward = 1.0 if self._correct else -0.1

        # Jump to outcome phase
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
            "task": "dms",
            "session_id": self._session_id,
            "trial_index": self._trial_index,
            "stimulus": self._test_stimulus,  # Main target stimulus is the test presentation
            "block_prior": None,
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
            "sample_stimulus": self._sample_stimulus,
            "delay_ms": self._delay_ms,
            "match": self._is_match,
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
            action = ACTION_NO_OP

        reward = 0.0
        if phase_name == "outcome" and self._phase_step == 0:
            reward = float(self._trial_reward)

        self._phase_step_counts[phase_name] += 1

        terminated = False
        truncated = False

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
            terminated = (self._trial_index + 1) >= self.config.trials_per_episode
            if terminated:
                self._terminated = True
                observation = self._terminal_observation()
                info = {
                    "session_id": self._session_id,
                    "trial_index": self._trial_index,
                    "phase": "terminal",
                    "phase_step": 0,
                    "sample_stimulus": self._sample_stimulus,
                    "delay_ms": self._delay_ms,
                    "match": self._is_match,
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
            obs["block_prior"] = np.zeros((), dtype=np.float32)
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
    "DMSConfig",
    "DelayedMatchToSampleEnv",
]
