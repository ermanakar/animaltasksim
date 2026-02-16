"""Bayesian observer baseline for AnimalTaskSim tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from animaltasksim.config import ProjectPaths
from animaltasksim.seeding import seed_everything
from envs.ibl_2afc import AgentMetadata as IBLAgentMetadata
from envs.ibl_2afc import IBL2AFCConfig, IBL2AFCEnv
from envs.rdm_macaque import AgentMetadata as RDMAgentMetadata
from envs.rdm_macaque import RDMConfig, RDMMacaqueEnv


@dataclass(slots=True)
class BayesParams:
    """Parameters governing the observer's decision rule."""

    sensory_sigma: float = 0.2
    lapse_rate: float = 0.02
    bias: float = 0.0

    def __post_init__(self) -> None:
        if self.sensory_sigma <= 0:
            raise ValueError("sensory_sigma must be positive")
        if not 0 <= self.lapse_rate < 0.5:
            raise ValueError("lapse_rate must be in [0, 0.5)")


@dataclass(slots=True)
class BayesTrainingConfig:
    """Training/evaluation configuration for the Bayesian observer."""

    env: Literal["ibl_2afc", "rdm"] = "ibl_2afc"
    episodes: int = 3
    trials_per_episode: int = 400
    seed: int = 1234
    agent_version: str = "0.1.0"
    output_dir: Path = field(default_factory=lambda: ProjectPaths.from_cwd().runs / "bayes")
    params: BayesParams = field(default_factory=BayesParams)

    def output_paths(self) -> dict[str, Path]:
        root = Path(self.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return {
            "root": root,
            "config": root / "config.json",
            "log": root / "trials.ndjson",
            "metrics": root / "evaluation.json",
        }


class BayesianObserver:
    """Stateless Bayesian observer acting in AnimalTaskSim environments."""

    def __init__(self, config: BayesTrainingConfig) -> None:
        self.config = config
        self.params = config.params
        self.rng = np.random.default_rng(config.seed)

    def _build_env(self, log_path: Path):
        if self.config.env == "ibl_2afc":
            env_config = IBL2AFCConfig(
                trials_per_episode=self.config.trials_per_episode,
                log_path=log_path,
                agent=IBLAgentMetadata(name="bayes_observer", version=self.config.agent_version),
                seed=self.config.seed,
            )
            env = IBL2AFCEnv(env_config)
            return env, env_config
        elif self.config.env == "rdm":
            env_config = RDMConfig(
                trials_per_episode=self.config.trials_per_episode,
                log_path=log_path,
                agent=RDMAgentMetadata(name="bayes_observer", version=self.config.agent_version),
                seed=self.config.seed,
            )
            env = RDMMacaqueEnv(env_config)
            return env, env_config
        raise ValueError(f"Unsupported environment: {self.config.env}")

    def _decide(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
        phase = info.get("phase")
        if self.config.env == "ibl_2afc":
            from envs.ibl_2afc import ACTION_LEFT, ACTION_NO_OP, ACTION_RIGHT

            if phase != "response":
                return ACTION_NO_OP
            raw = observation.get("contrast", 0.0)
            if isinstance(raw, np.ndarray):
                stimulus = float(raw.item())
            else:
                stimulus = float(raw)  # type: ignore[arg-type]
            return self._sample_binary_decision(stimulus, ACTION_LEFT, ACTION_RIGHT)
        else:
            from envs.rdm_macaque import ACTION_HOLD, ACTION_LEFT, ACTION_RIGHT

            if phase != "response":
                return ACTION_HOLD
            raw = observation.get("coherence", 0.0)
            if isinstance(raw, np.ndarray):
                stimulus = float(raw.item())
            else:
                stimulus = float(raw)  # type: ignore[arg-type]
            return self._sample_binary_decision(stimulus, ACTION_LEFT, ACTION_RIGHT)

    def _sample_binary_decision(self, stimulus: float, left_action: int, right_action: int) -> int:
        measurement = stimulus + self.rng.normal(0.0, self.params.sensory_sigma)
        measurement += self.params.bias
        base_action = right_action if measurement >= 0 else left_action
        if self.rng.random() < self.params.lapse_rate:
            return self.rng.choice([left_action, right_action])
        return base_action

    def run(self) -> dict[str, object]:
        paths = self.config.output_paths()
        env, env_config = self._build_env(paths["log"])

        seed_everything(self.config.seed)

        total_rewards: list[float] = []
        accuracy: list[float] = []

        for episode in range(self.config.episodes):
            observation, info = env.reset(seed=self.config.seed + episode)
            cumulative_reward = 0.0
            trial_rewards: list[float] = []
            correct = 0
            total = 0

            terminated = False
            while not terminated:
                action = self._decide(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward

                if info.get("phase") in {"iti", "terminal"}:
                    if reward > 0:
                        correct += 1
                    if reward >= 0:
                        trial_rewards.append(reward)
                        total += 1

            total_rewards.append(cumulative_reward)
            accuracy.append(correct / max(total, 1))

        env.close()

        metrics: dict[str, object] = {
            "episodes": self.config.episodes,
            "total_rewards": total_rewards,
            "mean_accuracy": accuracy,
        }
        paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._persist_config(paths["config"], env_config)
        return metrics

    def _persist_config(self, path: Path, env_config) -> None:
        payload = {
            "agent": {
                "name": "bayes_observer",
                "version": self.config.agent_version,
                "params": asdict(self.params),
            },
            "environment": self._serialize_env(env_config),
            "training": {
                "episodes": self.config.episodes,
                "trials_per_episode": self.config.trials_per_episode,
                "seed": self.config.seed,
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _serialize_env(self, env_config) -> dict[str, object]:
        if isinstance(env_config, IBL2AFCConfig):
            return {
                "trials_per_episode": env_config.trials_per_episode,
                "contrast_set": list(env_config.contrast_set),
                "step_ms": env_config.step_ms,
                "phase_schedule": [
                    {"name": phase.name, "duration_steps": phase.duration_steps}
                    for phase in env_config.phase_schedule
                ],
            }
        if isinstance(env_config, RDMConfig):
            return {
                "trials_per_episode": env_config.trials_per_episode,
                "coherence_set": list(env_config.coherence_set),
                "step_ms": env_config.step_ms,
                "phase_schedule": [
                    {"name": phase.name, "duration_steps": phase.duration_steps}
                    for phase in env_config.phase_schedule
                ],
                "per_step_cost": env_config.per_step_cost,
            }
        raise TypeError("Unsupported environment config")


def run_bayesian_observer(config: BayesTrainingConfig) -> dict[str, object]:
    observer = BayesianObserver(config)
    return observer.run()


__all__ = [
    "BayesParams",
    "BayesTrainingConfig",
    "BayesianObserver",
    "run_bayesian_observer",
]
