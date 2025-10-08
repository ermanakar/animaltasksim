"""Stimulus-sensitive logistic baseline (GLM) for the IBL 2AFC task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

from animaltasksim.config import ProjectPaths
from animaltasksim.seeding import seed_everything
from envs.ibl_2afc import (
    ACTION_LEFT,
    ACTION_NO_OP,
    ACTION_RIGHT,
    AgentMetadata,
    IBL2AFCConfig,
    IBL2AFCEnv,
)


@dataclass(slots=True)
class StickyGLMHyperParams:
    """Hyper-parameters for the GLM baseline."""

    learning_rate: float = 0.05
    weight_decay: float = 0.0
    temperature: float = 1.0
    sample_actions: bool = False  # if False, take argmax


@dataclass(slots=True)
class StickyGLMTrainingConfig:
    """Training configuration for the GLM baseline."""

    episodes: int = 10
    trials_per_episode: int = 400
    seed: int = 1234
    agent_version: str = "0.1.0"
    output_dir: Path = field(default_factory=lambda: ProjectPaths.from_cwd().runs / "ibl_sticky_glm")
    hyperparams: StickyGLMHyperParams = field(default_factory=StickyGLMHyperParams)

    def output_paths(self) -> dict[str, Path]:
        out = Path(self.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return {
            "root": out,
            "config": out / "config.json",
            "log": out / "trials.ndjson",
            "metrics": out / "training_metrics.json",
        }


class StickyGLMLearner:
    """Logistic policy with stickiness for contrast discrimination."""

    def __init__(self, config: StickyGLMTrainingConfig) -> None:
        self.config = config
        self.hyper = config.hyperparams
        self.weights = np.zeros(3, dtype=np.float64)
        self.temperature = max(self.hyper.temperature, 1e-6)

    def _features(self, contrast: float, prev_action_feature: float) -> np.ndarray:
        magnitude = abs(contrast)
        return np.array([contrast, prev_action_feature, magnitude], dtype=np.float64)

    @staticmethod
    def _label(contrast: float, block_prior: float) -> int:
        if contrast > 0:
            return 1
        if contrast < 0:
            return 0
        return 1 if block_prior >= 0.5 else 0

    def _prob_right(self, features: np.ndarray) -> float:
        logit = float(np.dot(self.weights, features)) / self.temperature
        if logit >= 0:
            z = np.exp(-logit)
            return 1.0 / (1.0 + z)
        z = np.exp(logit)
        return z / (1.0 + z)

    def _update(self, features: np.ndarray, label: int) -> None:
        prob = self._prob_right(features)
        grad = (prob - label) * features + self.hyper.weight_decay * self.weights
        self.weights -= self.hyper.learning_rate * grad

    def train(self) -> dict[str, object]:
        paths = self.config.output_paths()
        env_config = IBL2AFCConfig(
            trials_per_episode=self.config.trials_per_episode,
            log_path=paths["log"],
            agent=AgentMetadata(name="sticky_glm", version=self.config.agent_version),
            seed=self.config.seed,
        )
        env = IBL2AFCEnv(env_config)

        seed_everything(self.config.seed)

        episode_acc: list[float] = []
        total_rewards: list[float] = []

        prev_action_feature = 0.0

        for episode in range(self.config.episodes):
            observation, info = env.reset(seed=self.config.seed + episode)
            prev_action_feature = 0.0
            pending_features: np.ndarray | None = None
            pending_label: int | None = None
            accuracy = 0
            total = 0
            cumulative_reward = 0.0

            terminated = False
            while not terminated:
                phase = info["phase"]
                if phase == "response" and pending_features is None:
                    contrast = float(observation.get("contrast", 0.0))
                    block_prior = float(info.get("block_prior", 0.5))
                    features = self._features(contrast, prev_action_feature)
                    prob_right = self._prob_right(features)
                    if self.hyper.sample_actions:
                        action = ACTION_RIGHT if np.random.random() < prob_right else ACTION_LEFT
                    else:
                        action = ACTION_RIGHT if prob_right >= 0.5 else ACTION_LEFT
                    prev_action_feature = 1.0 if action == ACTION_RIGHT else -1.0
                    pending_features = features
                    pending_label = self._label(contrast, block_prior)
                else:
                    action = ACTION_NO_OP

                observation, reward, terminated, truncated, next_info = env.step(action)
                cumulative_reward += reward

                if pending_features is not None and next_info["phase"] == "outcome":
                    correct = bool(getattr(env, "_correct", reward > 0))
                    total += 1
                    if correct:
                        accuracy += 1
                    if pending_label is not None:
                        self._update(pending_features, pending_label)
                    pending_features = None
                    pending_label = None

                if pending_features is not None and next_info["phase"] in {"iti", "terminal"}:
                    if pending_label is not None:
                        self._update(pending_features, pending_label)
                    pending_features = None
                    pending_label = None

                observation, info = observation, next_info

            episode_acc.append(accuracy / max(total, 1))
            total_rewards.append(cumulative_reward)

        env.close()

        metrics = {
            "episodes": self.config.episodes,
            "mean_accuracy": episode_acc,
            "total_rewards": total_rewards,
            "weights": self.weights.tolist(),
        }
        paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._persist_config(paths["config"], env_config)
        return metrics

    def _persist_config(self, config_path: Path, env_config: IBL2AFCConfig) -> None:
        config_dict = {
            "agent": {
                "name": "sticky_glm",
                "version": self.config.agent_version,
                "hyperparams": asdict(self.hyper),
                "weights": self.weights.tolist(),
            },
            "environment": self._serialize_env_config(env_config),
            "training": {
                "episodes": self.config.episodes,
                "trials_per_episode": self.config.trials_per_episode,
                "seed": self.config.seed,
            },
        }
        config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    @staticmethod
    def _serialize_env_config(env_config: IBL2AFCConfig) -> dict[str, object]:
        return {
            "trials_per_episode": env_config.trials_per_episode,
            "contrast_set": list(env_config.contrast_set),
            "phase_schedule": [
                {"name": phase.name, "duration_steps": phase.duration_steps}
                for phase in env_config.phase_schedule
            ],
            "step_ms": env_config.step_ms,
            "expose_prior": env_config.expose_prior,
        }


def train_sticky_q(config: StickyGLMTrainingConfig) -> dict[str, object]:  # type: ignore[override]
    """Train the GLM baseline (kept under legacy name for CLI compatibility)."""

    trainer = StickyGLMLearner(config)
    return trainer.train()


__all__ = [
    "StickyGLMHyperParams",
    "StickyGLMLearner",
    "StickyGLMTrainingConfig",
    "train_sticky_q",
]
