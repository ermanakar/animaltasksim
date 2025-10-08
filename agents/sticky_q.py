"""Sticky-Q baseline agent for the IBL 2AFC task."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from animaltasksim.config import ProjectPaths
from animaltasksim.seeding import seed_everything
from envs.ibl_2afc import (
    ACTION_LEFT,
    ACTION_NO_OP,
    ACTION_RIGHT,
    ACTION_NAMES,
    AgentMetadata,
    BlockConfig,
    IBL2AFCConfig,
    IBL2AFCEnv,
)

StateKey = Tuple[int, int, int]


@dataclass(slots=True)
class StickyQHyperParams:
    """Tunable hyper-parameters for the Sticky-Q learner."""

    learning_rate: float = 0.1
    discount: float = 0.95
    epsilon: float = 0.2
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    stickiness: float = 1.0


@dataclass(slots=True)
class StickyQTrainingConfig:
    """Training configuration for the Sticky-Q agent."""

    episodes: int = 5
    trials_per_episode: int = 400
    seed: int = 1234
    agent_version: str = "0.1.0"
    output_dir: Path = field(default_factory=lambda: ProjectPaths.from_cwd().runs / "ibl_stickyq")
    hyperparams: StickyQHyperParams = field(default_factory=StickyQHyperParams)

    def output_paths(self) -> dict[str, Path]:
        out = Path(self.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return {
            "root": out,
            "config": out / "config.json",
            "log": out / "trials.ndjson",
            "metrics": out / "training_metrics.json",
        }


class StickyQLearner:
    """Tabular Q-learning agent with a stickiness bias."""

    def __init__(self, config: StickyQTrainingConfig) -> None:
        self.config = config
        self.hyper = config.hyperparams
        self.q_table: Dict[StateKey, np.ndarray] = {}
        self.epsilon = self.hyper.epsilon
        self.last_action_index: int | None = None

    def _encode_state(self, contrast: float, block_prior: float) -> StateKey:
        if contrast > 1e-6:
            stim = 1
        elif contrast < -1e-6:
            stim = -1
        else:
            stim = 0
        magnitude = abs(contrast)
        if magnitude >= 0.5:
            mag_bin = 2
        elif magnitude >= 0.25:
            mag_bin = 1
        else:
            mag_bin = 0
        bias = 1 if block_prior >= 0.5 else 0
        return (stim, mag_bin, bias)

    def _ensure_state(self, state: StateKey) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(2, dtype=np.float32)
        return self.q_table[state]

    def _select_action_index(self, state: StateKey) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 1)

        q_values = self._ensure_state(state)
        biased = q_values.copy()
        if self.last_action_index is not None:
            biased[self.last_action_index] += self.hyper.stickiness
        return int(np.argmax(biased))

    def _map_action_index_to_env(self, idx: int) -> int:
        return ACTION_LEFT if idx == 0 else ACTION_RIGHT

    def _update(self, state: StateKey, action_index: int, reward: float, next_state: StateKey | None) -> None:
        q_values = self._ensure_state(state)
        target = reward
        if next_state is not None:
            target += self.hyper.discount * float(np.max(self._ensure_state(next_state)))
        q_values[action_index] += self.hyper.learning_rate * (target - q_values[action_index])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.hyper.epsilon_min, self.epsilon * self.hyper.epsilon_decay)

    def train(self) -> dict[str, list[float]]:
        paths = self.config.output_paths()
        env_config = IBL2AFCConfig(
            trials_per_episode=self.config.trials_per_episode,
            log_path=paths["log"],
            agent=AgentMetadata(name="sticky_q", version=self.config.agent_version),
            seed=self.config.seed,
        )
        env = IBL2AFCEnv(env_config)

        seed_everything(self.config.seed)

        total_rewards: list[float] = []
        response_accuracy: list[float] = []

        for episode in range(self.config.episodes):
            obs, info = env.reset(seed=self.config.seed + episode)
            pending_state: StateKey | None = None
            pending_action_idx: int | None = None
            cumulative_reward = 0.0
            correct_trials = 0
            total_trials = 0

            terminated = False
            while not terminated:
                contrast = float(obs["contrast"])
                block_prior = float(info.get("block_prior", 0.5))
                state = self._encode_state(contrast, block_prior)

                phase = info["phase"]
                if phase == "response" and pending_action_idx is None:
                    action_idx = self._select_action_index(state)
                    action = self._map_action_index_to_env(action_idx)
                    pending_state = state
                    pending_action_idx = action_idx
                    self.last_action_index = action_idx
                else:
                    action = ACTION_NO_OP

                next_obs, reward, terminated, truncated, next_info = env.step(action)

                if phase == "response" and pending_action_idx is not None:
                    # Outcome evaluated on subsequent phase.
                    pass

                if pending_action_idx is not None and next_info["phase"] in {"outcome", "iti", "terminal"}:
                    next_contrast = float(next_obs["contrast"])
                    next_block_prior = float(next_info.get("block_prior", block_prior))
                    next_state = self._encode_state(next_contrast, next_block_prior)
                    terminal_transition = next_info["phase"] in {"iti", "terminal"}
                    self._update(pending_state, pending_action_idx, reward, None if terminal_transition else next_state)
                    if next_info["phase"] in {"outcome", "iti", "terminal"}:
                        is_correct = bool(getattr(env, "_correct", reward > 0))
                        if is_correct:
                            correct_trials += 1
                        total_trials += 1
                    pending_state = None
                    pending_action_idx = None
                    self.decay_epsilon()

                cumulative_reward += reward
                obs, info = next_obs, next_info

            if total_trials:
                response_accuracy.append(correct_trials / total_trials)
            total_rewards.append(cumulative_reward)

        env.close()

        metrics = {
            "episodes": self.config.episodes,
            "total_rewards": total_rewards,
            "response_accuracy": response_accuracy,
            "final_epsilon": self.epsilon,
        }

        paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._persist_config(paths["config"], env_config)
        return metrics

    def _persist_config(self, config_path: Path, env_config: IBL2AFCConfig) -> None:
        config_dict = {
            "agent": {
                "name": "sticky_q",
                "version": self.config.agent_version,
                "hyperparams": asdict(self.hyper),
            },
            "environment": self._serialize_env_config(env_config),
            "training": {
                "episodes": self.config.episodes,
                "trials_per_episode": self.config.trials_per_episode,
                "seed": self.config.seed,
            },
        }
        config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    def _serialize_env_config(self, env_config: IBL2AFCConfig) -> dict[str, object]:
        blocks = [asdict(block) for block in env_config.block_sequence]
        return {
            "trials_per_episode": env_config.trials_per_episode,
            "contrast_set": list(env_config.contrast_set),
            "block_sequence": blocks,
            "phase_schedule": [
                {"name": phase.name, "duration_steps": phase.duration_steps}
                for phase in env_config.phase_schedule
            ],
            "step_ms": env_config.step_ms,
            "include_phase_onehot": env_config.include_phase_onehot,
            "include_timing": env_config.include_timing,
            "expose_prior": env_config.expose_prior,
        }


def train_sticky_q(config: StickyQTrainingConfig) -> dict[str, list[float]]:
    """Train a Sticky-Q agent and return summary metrics."""

    trainer = StickyQLearner(config)
    return trainer.train()


__all__ = [
    "StickyQHyperParams",
    "StickyQLearner",
    "StickyQTrainingConfig",
    "train_sticky_q",
]
