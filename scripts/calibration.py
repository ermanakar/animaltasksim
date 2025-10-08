from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from agents.sticky_q import StickyQHyperParams, StickyQTrainingConfig, train_sticky_q
from agents.ppo_baseline import PPOHyperParams, PPOTrainingConfig, train_ppo
from eval.metrics import load_and_compute
from eval.report import build_report


@dataclass(slots=True)
class StickyOptions:
    output: Path = Path("runs/ibl_stickyq_calib")
    episodes: int = 15
    trials_per_episode: int = 400
    seed: int = 1234
    learning_rate: float = 0.1
    discount: float = 0.95
    epsilon: float = 0.4
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    stickiness: float = 0.1


@dataclass(slots=True)
class PPOOptions:
    output: Path = Path("runs/rdm_ppo_calib")
    total_timesteps: int = 40_000
    trials_per_episode: int = 400
    eval_episodes: int = 1
    seed: int = 1234
    per_step_cost: float = 0.01
    evidence_gain: float = 2.0
    momentary_sigma: float = 1.5
    include_cumulative_evidence: bool = True
    collapsing_bound: bool = False
    min_bound_steps: int = 5
    bound_threshold: float = 8.0
    learning_rate: float = 5e-5
    n_steps: int = 512
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5


@dataclass(slots=True)
class Args:
    mode: Literal["sticky", "ppo", "both"] = "both"
    sticky: StickyOptions = field(default_factory=StickyOptions)
    ppo: PPOOptions = field(default_factory=PPOOptions)
    verbose: bool = True


def _print_heading(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def _summarize_metrics(metrics: dict[str, object]) -> None:
    for section, values in metrics.items():
        if isinstance(values, dict):
            print(f"  {section}:")
            for key, value in values.items():
                if isinstance(value, float):
                    print(f"    {key:20s}: {value:.3f}")
                else:
                    print(f"    {key:20s}: {value}")


def run_sticky(opts: StickyOptions, verbose: bool = True) -> Path:
    hyper = StickyQHyperParams(
        learning_rate=opts.learning_rate,
        discount=opts.discount,
        epsilon=opts.epsilon,
        epsilon_min=opts.epsilon_min,
        epsilon_decay=opts.epsilon_decay,
        stickiness=opts.stickiness,
    )
    cfg = StickyQTrainingConfig(
        episodes=opts.episodes,
        trials_per_episode=opts.trials_per_episode,
        seed=opts.seed,
        output_dir=opts.output,
        hyperparams=hyper,
    )
    if verbose:
        _print_heading("Training Sticky-Q")
        print(json.dumps({"episodes": cfg.episodes, "trials_per_episode": cfg.trials_per_episode}, indent=2))
    train_sticky_q(cfg)

    paths = cfg.output_paths()
    metrics = load_and_compute(paths["log"])
    payload = {"log": str(paths["log"]), "metrics": metrics}
    metrics_path = paths["root"] / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = paths["root"] / "report.html"
    build_report(paths["log"], report_path, title="Sticky-Q Calibration", metrics=metrics)
    if verbose:
        _print_heading("Sticky-Q metrics")
        _summarize_metrics(metrics)
        print(f"Artifacts: {paths['root']}")
    return paths["root"]


def run_ppo(opts: PPOOptions, verbose: bool = True) -> Path:
    hyper = PPOHyperParams(
        learning_rate=opts.learning_rate,
        n_steps=opts.n_steps,
        batch_size=opts.batch_size,
        gamma=opts.gamma,
        gae_lambda=opts.gae_lambda,
        clip_range=opts.clip_range,
        ent_coef=opts.ent_coef,
        vf_coef=opts.vf_coef,
    )
    cfg = PPOTrainingConfig(
        env="rdm",
        total_timesteps=opts.total_timesteps,
        eval_trials=opts.trials_per_episode,
        eval_episodes=opts.eval_episodes,
        per_step_cost=opts.per_step_cost,
        evidence_gain=opts.evidence_gain,
        momentary_sigma=opts.momentary_sigma,
        include_cumulative_evidence=opts.include_cumulative_evidence,
        collapsing_bound=opts.collapsing_bound,
        min_bound_steps=opts.min_bound_steps,
        bound_threshold=opts.bound_threshold,
        seed=opts.seed,
        output_dir=opts.output,
        hyperparams=hyper,
    )
    if verbose:
        _print_heading("Training PPO")
        print(json.dumps({
            "total_timesteps": cfg.total_timesteps,
            "trials_per_episode": cfg.eval_trials,
            "per_step_cost": cfg.per_step_cost,
        }, indent=2))
    train_ppo(cfg)

    paths = cfg.output_paths()
    metrics = load_and_compute(paths["log"])
    payload = {"log": str(paths["log"]), "metrics": metrics}
    metrics_path = paths["root"] / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = paths["root"] / "report.html"
    build_report(paths["log"], report_path, title="PPO Calibration", metrics=metrics)
    if verbose:
        _print_heading("PPO metrics")
        _summarize_metrics(metrics)
        print(f"Artifacts: {paths['root']}")
    return paths["root"]


def main() -> None:
    args = tyro.cli(Args)
    if args.mode in {"sticky", "both"}:
        run_sticky(args.sticky, verbose=args.verbose)
    if args.mode in {"ppo", "both"}:
        run_ppo(args.ppo, verbose=args.verbose)


if __name__ == "__main__":
    main()
