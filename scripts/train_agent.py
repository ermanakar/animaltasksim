from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from agents.bayes_observer import BayesParams, BayesTrainingConfig, run_bayesian_observer
from agents.ppo_baseline import PPOHyperParams, PPOTrainingConfig, train_ppo
from agents.sticky_q import StickyGLMHyperParams, StickyGLMTrainingConfig, train_sticky_q


@dataclass(slots=True)
class StickyQCLIConfig:
    learning_rate: float = 0.05
    weight_decay: float = 0.0
    temperature: float = 1.0
    sample_actions: bool = False


@dataclass(slots=True)
class BayesCLIConfig:
    sensory_sigma: float = 0.2
    lapse_rate: float = 0.02
    bias: float = 0.0


@dataclass(slots=True)
class PPOCLIConfig:
    learning_rate: float = 5e-5
    n_steps: int = 512
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    per_step_cost: float = 0.02
    evidence_gain: float = 0.05
    momentary_sigma: float = 1.0
    include_cumulative_evidence: bool = True
    collapsing_bound: bool = True
    min_bound_steps: int = 5
    bound_threshold: float = 3.0
    use_confidence_reward: bool = False
    include_history: bool = False  # For RDM
    ibl_include_history: bool = False  # For IBL
    intratrial_evidence_schedule: tuple[float, ...] = ()


@dataclass(slots=True)
class TrainArgs:
    env: Literal["ibl_2afc", "rdm"] = "ibl_2afc"
    agent: Literal["sticky_q", "bayes", "ppo"] = "sticky_q"
    steps: int = 400
    episodes: int | None = None
    trials_per_episode: int = 400
    seed: int = 1234
    out: Path = Path("runs/ibl_stickyq")
    agent_version: str = "0.1.0"
    sticky_q: StickyQCLIConfig = field(default_factory=StickyQCLIConfig)
    bayes: BayesCLIConfig = field(default_factory=BayesCLIConfig)
    ppo: PPOCLIConfig = field(default_factory=PPOCLIConfig)


def _train_sticky_q(args: TrainArgs) -> dict[str, object]:
    if args.env != "ibl_2afc":
        raise NotImplementedError("Sticky-Q is only configured for ibl_2afc in this release.")

    episodes = args.episodes
    if episodes is None:
        episodes = max(1, math.ceil(args.steps / max(1, args.trials_per_episode)))

    hyper = StickyGLMHyperParams(**asdict(args.sticky_q))
    config = StickyGLMTrainingConfig(
        episodes=episodes,
        trials_per_episode=args.trials_per_episode,
        seed=args.seed,
        agent_version=args.agent_version,
        output_dir=args.out,
        hyperparams=hyper,
    )
    return train_sticky_q(config)


def _train_bayes(args: TrainArgs) -> dict[str, object]:
    episodes = args.episodes
    if episodes is None:
        episodes = max(1, math.ceil(args.steps / max(1, args.trials_per_episode)))

    params = BayesParams(**asdict(args.bayes))
    config = BayesTrainingConfig(
        env=args.env,
        episodes=episodes,
        trials_per_episode=args.trials_per_episode,
        seed=args.seed,
        agent_version=args.agent_version,
        output_dir=args.out,
        params=params,
    )
    return run_bayesian_observer(config)


def _train_ppo(args: TrainArgs) -> dict[str, object]:
    ppo_args = asdict(args.ppo)
    ibl_include_history = bool(ppo_args.pop("ibl_include_history"))
    per_step_cost = float(ppo_args.pop("per_step_cost"))
    evidence_gain = float(ppo_args.pop("evidence_gain"))
    momentary_sigma = float(ppo_args.pop("momentary_sigma"))
    include_cumulative_evidence = ppo_args.pop("include_cumulative_evidence")
    collapsing_bound = ppo_args.pop("collapsing_bound")
    min_bound_steps = int(ppo_args.pop("min_bound_steps"))
    bound_threshold = float(ppo_args.pop("bound_threshold"))
    use_confidence_reward = bool(ppo_args.pop("use_confidence_reward"))
    include_history = bool(ppo_args.pop("include_history"))
    intratrial_evidence_schedule = tuple(float(v) for v in ppo_args.pop("intratrial_evidence_schedule"))
    hyper = PPOHyperParams(**ppo_args)
    config = PPOTrainingConfig(
        env=args.env,
        total_timesteps=max(args.steps, 1),
        eval_trials=args.trials_per_episode,
        eval_episodes=args.episodes or 1,
        per_step_cost=per_step_cost,
        evidence_gain=evidence_gain,
        momentary_sigma=momentary_sigma,
        include_cumulative_evidence=include_cumulative_evidence,
        collapsing_bound=collapsing_bound,
        min_bound_steps=min_bound_steps,
        bound_threshold=bound_threshold,
        use_confidence_reward=use_confidence_reward,
        include_history=include_history,
        ibl_include_history=ibl_include_history,
        intratrial_evidence_schedule=intratrial_evidence_schedule,
        seed=args.seed,
        agent_version=args.agent_version,
        output_dir=args.out,
        hyperparams=hyper,
    )
    return train_ppo(config)


def main(args: TrainArgs) -> None:
    if args.agent == "sticky_q":
        metrics = _train_sticky_q(args)
    elif args.agent == "bayes":
        metrics = _train_bayes(args)
    elif args.agent == "ppo":
        metrics = _train_ppo(args)
    else:
        raise NotImplementedError("Unknown agent selection")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
