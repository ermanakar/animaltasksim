"""Macaque random-dot motion (RDM) environment with streaming evidence."""

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
ACTION_HOLD = 2
ACTION_NAMES = {
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
    ACTION_HOLD: "hold",
}

STREAMING_PHASES = {"stimulus", "response"}

DEFAULT_PHASE_SCHEDULE = (
    PhaseTiming("fixation", 10),
    PhaseTiming("stimulus", 80),
    PhaseTiming("response", 120),
    PhaseTiming("outcome", 10),
)


def _validate_coherences(values: Iterable[float]) -> list[float]:
    coherences = [float(v) for v in values]
    if not coherences:
        raise ValueError("coherence_set must contain at least one value")
    for value in coherences:
        if value < 0.0 or value > 1.0:
            raise ValueError("coherence magnitudes must be within [0, 1]")
    return coherences


@dataclass(slots=True)
class AgentMetadata:
    name: str = "unknown"
    version: str = "0.0"

    def to_dict(self) -> dict[str, str]:
        if not self.name or not self.version:
            raise ValueError("agent metadata requires non-empty name/version")
        return {"name": self.name, "version": self.version}


@dataclass(slots=True)
class RDMConfig:
    trials_per_episode: int = 300
    coherence_set: tuple[float, ...] = (0.0, 0.032, 0.064, 0.128, 0.256, 0.512)  # Match Roitman coherences
    phase_schedule: tuple[PhaseTiming, ...] = DEFAULT_PHASE_SCHEDULE
    step_ms: int = 10
    include_phase_onehot: bool = False
    include_timing: bool = False
    include_cumulative_evidence: bool = True
    evidence_gain: float = 0.05
    momentary_sigma: float = 1.0
    per_step_cost: float = 0.0
    collapsing_bound: bool = False
    min_bound_steps: int = 20  # 200ms minimum RT (matches Roitman data)
    bound_threshold: float = 3.0
    # Confidence-based reward shaping parameters
    use_confidence_reward: bool = False  # Enable new reward structure
    confidence_bonus_weight: float = 1.0  # Multiplier for confidence bonus
    base_time_cost: float = 0.0001  # Base per-step cost
    time_cost_growth: float = 0.01  # How much time cost grows with steps
    # RT shaping: reward RTs in realistic range (target 400-800ms)
    target_rt_steps: int = 60  # ~600ms at 10ms/step
    rt_tolerance: float = 30.0  # Gaussian width for RT bonus
    log_path: Path | None = None
    agent: AgentMetadata = field(default_factory=AgentMetadata)
    seed: int | None = None

    def __post_init__(self) -> None:
        ensure_phase_names(self.phase_schedule)
        if self.trials_per_episode <= 0:
            raise ValueError("trials_per_episode must be positive")
        if self.step_ms <= 0:
            raise ValueError("step_ms must be positive")
        _validate_coherences(self.coherence_set)
        if self.collapsing_bound and self.min_bound_steps <= 0:
            raise ValueError("min_bound_steps must be positive when collapsing_bound is enabled")
        if self.momentary_sigma <= 0:
            raise ValueError("momentary_sigma must be positive")
        if self.bound_threshold <= 0:
            raise ValueError("bound_threshold must be positive")


class RDMMacaqueEnv(Env):
    """Gymnasium environment for the macaque RDM decision task."""

    metadata = {"render_modes": []}

    def __init__(self, config: RDMConfig | None = None) -> None:
        super().__init__()
        self.config = config or RDMConfig()
        self._phase_schedule = self.config.phase_schedule
        self._phase_names = ensure_phase_names(self._phase_schedule)
        self._coherences = _validate_coherences(self.config.coherence_set)

        self.action_space = spaces.Discrete(3)
        self.observation_space = self._build_observation_space()

        self._logger: NDJSONTrialLogger | None = None
        if self.config.log_path is not None:
            self._logger = NDJSONTrialLogger(self.config.log_path)

        self._rng: np.random.Generator | None = None
        self._seed: int | None = None
        self._session_id: str = ""
        self._trial_index: int = 0
        self._terminated: bool = False
        self._phase_index: int = 0
        self._phase_step: int = 0
        self._phase_step_counts: dict[str, int] = {}
        self._stimulus: dict[str, object] = {}
        self._response_action: str = ACTION_NAMES[ACTION_HOLD]
        self._response_captured: bool = False
        self._trial_reward: float = 0.0
        self._correct: bool = False
        self._rt_steps: int | None = None
        self._prev_action: str | None = None
        self._prev_reward: float | None = None
        self._prev_correct: bool | None = None
        self._signed_coherence: float = 0.0
        self._momentary_evidence: float = 0.0
        self._cumulative_evidence: float = 0.0
        self._response_steps: int = 0

    def _build_observation_space(self) -> spaces.Dict:
        space_dict: dict[str, spaces.Space] = {
            "coherence": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
        }
        if self.config.include_cumulative_evidence:
            space_dict["cumulative_evidence"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(), dtype=np.float32
            )
        if self.config.include_phase_onehot:
            space_dict["phase_onehot"] = spaces.Box(
                low=0.0, high=1.0, shape=(len(self._phase_schedule),), dtype=np.float32
            )
        if self.config.include_timing:
            space_dict["t_norm"] = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        return spaces.Dict(space_dict)

    @property
    def _current_phase(self) -> PhaseTiming:
        return self._phase_schedule[self._phase_index]

    def _current_phase_name(self) -> str:
        return self._current_phase.name

    def _reset_evidence(self) -> None:
        self._momentary_evidence = 0.0
        self._cumulative_evidence = 0.0

    def _sample_evidence(self) -> None:
        mean = self.config.evidence_gain * self._signed_coherence
        sample = float(self._rng.normal(loc=mean, scale=self.config.momentary_sigma)) if self._rng else 0.0
        self._momentary_evidence = sample
        self._cumulative_evidence += sample

    def _maybe_sample_evidence(self) -> None:
        phase_name = self._current_phase_name()
        if phase_name in STREAMING_PHASES:
            self._sample_evidence()
        else:
            self._momentary_evidence = 0.0

    def _build_observation(self) -> dict[str, np.ndarray | float | np.floating]:  # type: ignore[return]
        obs: dict[str, np.ndarray | float | np.floating] = {  # type: ignore[misc]
            "coherence": np.float32(self._momentary_evidence)
        }
        if self.config.include_cumulative_evidence:
            obs["cumulative_evidence"] = np.float32(self._cumulative_evidence)
        if self.config.include_phase_onehot:
            onehot = np.zeros(len(self._phase_schedule), dtype=np.float32)
            onehot[self._phase_index] = 1.0
            obs["phase_onehot"] = onehot
        if self.config.include_timing:
            duration = self._current_phase.duration_steps
            obs["t_norm"] = np.float32(self._phase_step / max(duration - 1, 1))
        return obs

    def _default_info(self) -> dict[str, object]:
        return {
            "session_id": self._session_id,
            "trial_index": self._trial_index,
            "phase": self._current_phase_name(),
            "phase_step": self._phase_step,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        s = seed if seed is not None else self.config.seed
        self._rng, actual_seed = seeding.np_random(s)
        self._seed = int(actual_seed)
        options = options or {}
        self._session_id = options.get("session_id", str(uuid4()))

        self._trial_index = 0
        self._terminated = False
        self._phase_index = 0
        self._phase_step = 0
        self._phase_step_counts = {name: 0 for name in self._phase_names}
        self._response_action = ACTION_NAMES[ACTION_HOLD]
        self._response_captured = False
        self._trial_reward = 0.0
        self._correct = False
        self._rt_steps = None
        self._prev_action = None
        self._prev_reward = None
        self._prev_correct = None
        self._response_steps = 0

        self._start_new_trial()
        self._maybe_sample_evidence()
        return self._build_observation(), self._default_info()

    def _start_new_trial(self) -> None:
        magnitude = float(self._rng.choice(self._coherences)) if self._rng else 0.0
        direction_right = self._rng.random() < 0.5 if self._rng else True
        sign = 1.0 if direction_right else -1.0
        coherence_value = magnitude * sign
        if magnitude == 0.0:
            coherence_value = 0.0
        self._signed_coherence = coherence_value
        self._stimulus = {
            "coherence": coherence_value,
            "direction": "right" if coherence_value > 0 else "left" if coherence_value < 0 else "none",
        }
        self._phase_index = 0
        self._phase_step = 0
        self._phase_step_counts = {name: 0 for name in self._phase_names}
        self._response_action = ACTION_NAMES[ACTION_HOLD]
        self._response_captured = False
        self._trial_reward = 0.0
        self._correct = False
        self._rt_steps = None
        self._reset_evidence()
        self._response_steps = 0

    def _log_trial(self) -> None:
        if self._logger is None:
            return
        phase_times = {
            f"{name}_ms": self.config.step_ms * steps
            for name, steps in self._phase_step_counts.items()
        }
        record = {
            "task": "rdm",
            "session_id": self._session_id,
            "trial_index": self._trial_index,
            "stimulus": self._stimulus,
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
        }
        self._logger.log(record)
        self._prev_action = self._response_action
        self._prev_reward = float(self._trial_reward)
        self._prev_correct = self._correct

    def _apply_per_step_cost(self, phase_name: str) -> float:
        return 0.0

    def _compute_confidence_based_reward(self, correct: bool, response_steps: int) -> float:
        """
        Compute reward with confidence bonus and adaptive time cost.
        
        Encourages agent to:
        - Build strong evidence before committing (confidence bonus)
        - Commit quickly when evidence is strong (high coherence)
        - Wait longer when evidence is weak (low coherence)
        - Not wait forever (time cost grows with steps)
        """
        # Base reward for correctness
        base_reward = 1.0 if correct else 0.0
        
        # Confidence = strength of cumulative evidence at commit time
        # Higher coherence → faster accumulation → higher confidence for same wait time
        max_steps = self._phase_schedule[2].duration_steps  # response phase duration
        max_possible_evidence = max_steps * self.config.evidence_gain * 0.512  # highest coherence
        confidence = abs(self._cumulative_evidence) / max(max_possible_evidence, 0.01)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Confidence bonus (only rewarded if correct)
        confidence_bonus = 0.0
        if correct:
            confidence_bonus = self.config.confidence_bonus_weight * confidence
        
        # Adaptive time cost: cheap early, expensive late
        # This creates urgency without forcing premature commits
        time_cost = self.config.base_time_cost * response_steps * (1.0 + self.config.time_cost_growth * response_steps)
        
        # RT shaping bonus: reward RTs near target (e.g., 600ms = 60 steps)
        # Gaussian bonus peaking at target_rt_steps with width rt_tolerance
        rt_bonus = 0.0
        if correct and self.config.target_rt_steps > 0 and self.config.rt_tolerance > 0:
            rt_deviation = abs(response_steps - self.config.target_rt_steps)
            rt_bonus = 0.5 * np.exp(-(rt_deviation ** 2) / (2 * self.config.rt_tolerance ** 2))
        
        reward = base_reward * (1.0 + confidence_bonus + rt_bonus) - time_cost
        return float(reward)

    def _process_response(self, action: int) -> None:
        if self._response_captured:
            return
        if action not in ACTION_NAMES:
            raise ValueError(f"invalid action {action}")
        if action == ACTION_HOLD:
            return
        # Enforce a minimum evidence accumulation period before allowing commits.
        response_steps = self._phase_step_counts.get("response", 0)
        if response_steps < self.config.min_bound_steps:
            return

        self._response_captured = True
        self._response_action = ACTION_NAMES[action]
        self._rt_steps = self._phase_step + 1

        expected = ACTION_RIGHT if self._signed_coherence > 0 else ACTION_LEFT if self._signed_coherence < 0 else ACTION_RIGHT
        self._correct = action == expected
        
        # Use confidence-based reward if enabled, otherwise simple correct/incorrect
        if self.config.use_confidence_reward:
            self._trial_reward = self._compute_confidence_based_reward(self._correct, response_steps)
        else:
            self._trial_reward = 1.0 if self._correct else -1.0

        self._phase_step = self._current_phase.duration_steps - 1

    def _finalize_without_response(self) -> None:
        if self._response_captured:
            return
        self._response_action = ACTION_NAMES[ACTION_HOLD]
        self._response_captured = True
        self._trial_reward = 0.0
        self._rt_steps = None
        self._correct = False

    def step(self, action: int):  # type: ignore[override]
        if self._terminated:
            raise RuntimeError("Episode already terminated; call reset().")

        self._maybe_sample_evidence()

        phase_name = self._current_phase_name()
        reward = 0.0

        if phase_name == "response":
            if not self._response_captured:
                self._response_steps += 1
            self._process_response(int(action))
        elif phase_name in {"fixation", "stimulus", "outcome"}:
            action = ACTION_HOLD
        else:
            action = ACTION_HOLD

        reward += self._apply_per_step_cost(phase_name)

        if phase_name == "outcome" and self._phase_step == 0:
            reward = self._trial_reward
            if self.config.per_step_cost > 0.0:
                reward -= self.config.per_step_cost * self._response_steps
            self._response_steps = 0

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

        if (
            self.config.collapsing_bound
            and not self._response_captured
            and phase_name == "response"
            and self._phase_step_counts.get("response", 0) >= self.config.min_bound_steps
        ):
            response_steps = self._phase_step_counts.get("response", 0)
            dynamic_bound = max(0.5, self.config.bound_threshold * np.exp(-0.02 * response_steps))
            if abs(self._cumulative_evidence) >= dynamic_bound:
                action = ACTION_RIGHT if self._cumulative_evidence >= 0 else ACTION_LEFT
                self._process_response(action)

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
                }
            else:
                self._trial_index += 1
                self._start_new_trial()
                self._maybe_sample_evidence()
                observation = self._build_observation()
                info = self._default_info()
        else:
            observation = self._build_observation()
            info = self._default_info()

        return observation, reward, terminated, truncated, info

    def _terminal_observation(self) -> dict[str, np.ndarray | float | np.floating]:  # type: ignore[return]
        obs: dict[str, np.ndarray | float | np.floating] = {"coherence": np.float32(0.0)}  # type: ignore[misc]
        if self.config.include_cumulative_evidence:
            obs["cumulative_evidence"] = np.float32(0.0)
        if self.config.include_phase_onehot:
            obs["phase_onehot"] = np.zeros(len(self._phase_schedule), dtype=np.float32)
        if self.config.include_timing:
            obs["t_norm"] = np.float32(0.0)
        return obs

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()
        super().close()


__all__ = [
    "ACTION_HOLD",
    "ACTION_LEFT",
    "ACTION_NAMES",
    "ACTION_RIGHT",
    "AgentMetadata",
    "RDMConfig",
    "RDMMacaqueEnv",
]
