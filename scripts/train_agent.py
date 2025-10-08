from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from agents.bayes_observer import BayesParams, BayesTrainingConfig, run_bayesian_observer
from agents.ppo_baseline import PPOHyperParams, PPOTrainingConfig, train_ppo
from agents.sticky_q import StickyQHyperParams, StickyQTrainingConfig, train_sticky_q


@dataclass(slots=True)
class StickyQCLIConfig:
    learning_rate: float = 0.1
    discount: float = 0.95
    epsilon: float = 0.2
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    stickiness: float = 1.0


@dataclass(slots=True)
class BayesCLIConfig:
    sensory_sigma: float = 0.2
    lapse_rate: float = 0.02
    bias: float = 0.0


@dataclass(slots=True)
class PPOCLIConfig:
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    per_step_cost: float = 0.0


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

    hyper = StickyQHyperParams(**asdict(args.sticky_q))
    config = StickyQTrainingConfig(
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
    per_step_cost = float(ppo_args.pop("per_step_cost"))
    hyper = PPOHyperParams(**ppo_args)
    config = PPOTrainingConfig(
        env=args.env,
        total_timesteps=max(args.steps, 1),
        eval_trials=args.trials_per_episode,
        eval_episodes=args.episodes or 1,
        per_step_cost=per_step_cost,
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
