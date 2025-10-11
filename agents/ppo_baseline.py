"""PPO baseline agent with action masking for AnimalTaskSim tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

from animaltasksim.config import ProjectPaths
from animaltasksim.seeding import seed_everything
from envs.ibl_2afc import AgentMetadata as IBLAgentMetadata
from envs.ibl_2afc import IBL2AFCConfig, IBL2AFCEnv
from envs.rdm_macaque import AgentMetadata as RDMAgentMetadata
from envs.rdm_macaque import RDMConfig, RDMMacaqueEnv


@dataclass(slots=True)
class PPOHyperParams:
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5


@dataclass(slots=True)
class PPOTrainingConfig:
    env: Literal["ibl_2afc", "rdm"] = "ibl_2afc"
    total_timesteps: int = 60_000
    eval_trials: int = 600
    eval_episodes: int = 1
    per_step_cost: float = 0.02
    evidence_gain: float = 0.05
    momentary_sigma: float = 1.0
    include_cumulative_evidence: bool = True
    collapsing_bound: bool = False
    min_bound_steps: int = 20
    bound_threshold: float = 3.0
    # Confidence-based reward parameters
    use_confidence_reward: bool = False
    confidence_bonus_weight: float = 1.0
    base_time_cost: float = 0.0001
    time_cost_growth: float = 0.01
    target_rt_steps: int = 60
    rt_tolerance: float = 30.0
    use_avg_reward_time_cost: bool = False
    avg_reward_alpha: float = 0.05
    avg_reward_scale: float = 1.0
    avg_reward_initial_rate: float = 1.0
    include_urgency_feature: bool = False
    urgency_slope: float = 1.0
    seed: int = 1234
    agent_version: str = "0.1.0"
    output_dir: Path = field(default_factory=lambda: ProjectPaths.from_cwd().runs / "ppo")
    hyperparams: PPOHyperParams = field(default_factory=PPOHyperParams)

    def output_paths(self) -> dict[str, Path]:
        root = Path(self.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return {
            "root": root,
            "config": root / "config.json",
            "log": root / "trials.ndjson",
            "metrics": root / "evaluation.json",
            "model": root / "model.zip",
        }


def _make_env(
    env_name: str,
    *,
    trials: int,
    seed: int,
    log_path: Path | None,
    per_step_cost: float,
    agent_version: str,
    evidence_gain: float,
    momentary_sigma: float,
    include_cumulative_evidence: bool,
    collapsing_bound: bool,
    min_bound_steps: int,
    bound_threshold: float,
    use_confidence_reward: bool = False,
    confidence_bonus_weight: float = 1.0,
    base_time_cost: float = 0.0001,
    time_cost_growth: float = 0.01,
    target_rt_steps: int = 60,
    rt_tolerance: float = 30.0,
    use_avg_reward_time_cost: bool = False,
    avg_reward_alpha: float = 0.05,
    avg_reward_scale: float = 1.0,
    avg_reward_initial_rate: float = 1.0,
    include_urgency_feature: bool = False,
    urgency_slope: float = 1.0,
):
    if env_name == "ibl_2afc":
        config = IBL2AFCConfig(
            trials_per_episode=trials,
            log_path=log_path,
            agent=IBLAgentMetadata(name="ppo_baseline", version=agent_version),
            seed=seed,
        )
        env = IBL2AFCEnv(config)
    elif env_name == "rdm":
        config = RDMConfig(
            trials_per_episode=trials,
            per_step_cost=per_step_cost,
            log_path=log_path,
            agent=RDMAgentMetadata(name="ppo_baseline", version=agent_version),
            seed=seed,
            evidence_gain=evidence_gain,
            momentary_sigma=momentary_sigma,
            include_cumulative_evidence=include_cumulative_evidence,
            collapsing_bound=collapsing_bound,
            min_bound_steps=min_bound_steps,
            bound_threshold=bound_threshold,
            use_confidence_reward=use_confidence_reward,
            confidence_bonus_weight=confidence_bonus_weight,
            base_time_cost=base_time_cost,
            time_cost_growth=time_cost_growth,
            target_rt_steps=target_rt_steps,
            rt_tolerance=rt_tolerance,
            use_avg_reward_time_cost=use_avg_reward_time_cost,
            avg_reward_alpha=avg_reward_alpha,
            avg_reward_scale=avg_reward_scale,
            avg_reward_initial_rate=avg_reward_initial_rate,
            include_urgency_feature=include_urgency_feature,
            urgency_slope=urgency_slope,
        )
        env = RDMMacaqueEnv(config)
    else:
        raise ValueError(f"Unsupported environment {env_name}")

    return FlattenObservation(env), config


def _evaluate_policy(model: PPO, env: gym.Env, episodes: int) -> dict[str, list[float]]:
    rewards_per_episode: list[float] = []
    lengths: list[float] = []

    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rewards_per_episode.append(total_reward)
        lengths.append(float(steps))
    return {"reward": rewards_per_episode, "length": lengths}


def train_ppo(config: PPOTrainingConfig) -> dict[str, object]:
    paths = config.output_paths()
    seed_everything(config.seed)

    train_env, train_env_config = _make_env(
        config.env,
        trials=config.eval_trials,
        seed=config.seed,
        log_path=None,
        per_step_cost=config.per_step_cost,
        agent_version=config.agent_version,
        evidence_gain=config.evidence_gain,
        momentary_sigma=config.momentary_sigma,
        include_cumulative_evidence=config.include_cumulative_evidence,
        collapsing_bound=config.collapsing_bound,
        min_bound_steps=config.min_bound_steps,
        bound_threshold=config.bound_threshold,
        use_confidence_reward=config.use_confidence_reward,
        confidence_bonus_weight=config.confidence_bonus_weight,
        base_time_cost=config.base_time_cost,
        time_cost_growth=config.time_cost_growth,
        target_rt_steps=config.target_rt_steps,
        rt_tolerance=config.rt_tolerance,
        use_avg_reward_time_cost=config.use_avg_reward_time_cost,
        avg_reward_alpha=config.avg_reward_alpha,
        avg_reward_scale=config.avg_reward_scale,
        avg_reward_initial_rate=config.avg_reward_initial_rate,
        include_urgency_feature=config.include_urgency_feature,
        urgency_slope=config.urgency_slope,
    )

    hyper = config.hyperparams
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=hyper.learning_rate,
        n_steps=hyper.n_steps,
        batch_size=hyper.batch_size,
        gamma=hyper.gamma,
        gae_lambda=hyper.gae_lambda,
        clip_range=hyper.clip_range,
        ent_coef=hyper.ent_coef,
        vf_coef=hyper.vf_coef,
        seed=config.seed,
        verbose=0,
    )

    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    model.save(paths["model"])
    train_env.close()

    eval_env, eval_env_config = _make_env(
        config.env,
        trials=config.eval_trials,
        seed=config.seed + 10,
        log_path=paths["log"],
        per_step_cost=config.per_step_cost,
        agent_version=config.agent_version,
        evidence_gain=config.evidence_gain,
        momentary_sigma=config.momentary_sigma,
        include_cumulative_evidence=config.include_cumulative_evidence,
        collapsing_bound=config.collapsing_bound,
        min_bound_steps=config.min_bound_steps,
        bound_threshold=config.bound_threshold,
        use_confidence_reward=config.use_confidence_reward,
        confidence_bonus_weight=config.confidence_bonus_weight,
        base_time_cost=config.base_time_cost,
        time_cost_growth=config.time_cost_growth,
        target_rt_steps=config.target_rt_steps,
        rt_tolerance=config.rt_tolerance,
        use_avg_reward_time_cost=config.use_avg_reward_time_cost,
        avg_reward_alpha=config.avg_reward_alpha,
        avg_reward_scale=config.avg_reward_scale,
        avg_reward_initial_rate=config.avg_reward_initial_rate,
        include_urgency_feature=config.include_urgency_feature,
        urgency_slope=config.urgency_slope,
    )

    evaluation = _evaluate_policy(model, eval_env, episodes=config.eval_episodes)
    eval_env.close()

    metrics_payload = {
        "episodes": config.eval_episodes,
        "total_reward": evaluation["reward"],
        "episode_length": evaluation["length"],
    }
    paths["metrics"].write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    _persist_config(paths["config"], config, train_env_config, eval_env_config)
    return metrics_payload


def _persist_config(path: Path, config: PPOTrainingConfig, train_env_config, eval_env_config) -> None:
    payload = {
        "agent": {
            "name": "ppo_baseline",
            "version": config.agent_version,
            "hyperparams": asdict(config.hyperparams),
            "total_timesteps": config.total_timesteps,
        },
        "environment": {
            "train": _serialize_env(train_env_config),
            "eval": _serialize_env(eval_env_config),
        },
        "seed": config.seed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _serialize_env(env_config) -> dict[str, object]:
    if isinstance(env_config, IBL2AFCConfig):
        return {
            "trials_per_episode": env_config.trials_per_episode,
            "contrast_set": list(env_config.contrast_set),
            "phase_schedule": [
                {"name": phase.name, "duration_steps": phase.duration_steps}
                for phase in env_config.phase_schedule
            ],
            "step_ms": env_config.step_ms,
        }
    if isinstance(env_config, RDMConfig):
        return {
            "trials_per_episode": env_config.trials_per_episode,
            "coherence_set": list(env_config.coherence_set),
            "phase_schedule": [
                {"name": phase.name, "duration_steps": phase.duration_steps}
                for phase in env_config.phase_schedule
            ],
            "step_ms": env_config.step_ms,
            "per_step_cost": env_config.per_step_cost,
            "evidence_gain": env_config.evidence_gain,
            "momentary_sigma": env_config.momentary_sigma,
            "include_cumulative_evidence": env_config.include_cumulative_evidence,
            "collapsing_bound": env_config.collapsing_bound,
            "min_bound_steps": env_config.min_bound_steps,
            "bound_threshold": env_config.bound_threshold,
            "use_confidence_reward": env_config.use_confidence_reward,
            "confidence_bonus_weight": env_config.confidence_bonus_weight,
            "base_time_cost": env_config.base_time_cost,
            "time_cost_growth": env_config.time_cost_growth,
            "target_rt_steps": env_config.target_rt_steps,
            "rt_tolerance": env_config.rt_tolerance,
            "use_avg_reward_time_cost": env_config.use_avg_reward_time_cost,
            "avg_reward_alpha": env_config.avg_reward_alpha,
            "avg_reward_scale": env_config.avg_reward_scale,
            "avg_reward_initial_rate": env_config.avg_reward_initial_rate,
            "include_urgency_feature": env_config.include_urgency_feature,
            "urgency_slope": env_config.urgency_slope,
        }
    raise TypeError("Unsupported environment configuration")


__all__ = [
    "PPOHyperParams",
    "PPOTrainingConfig",
    "train_ppo",
]
