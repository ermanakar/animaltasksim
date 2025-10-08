from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from agents.ddm_baseline import DDMConfig, DDMBaseline
from agents.sticky_q import StickyGLMHyperParams, StickyGLMTrainingConfig, train_sticky_q
from agents.ppo_baseline import PPOHyperParams, PPOTrainingConfig, train_ppo
from eval.metrics import load_and_compute
from eval.report import build_report


@dataclass(slots=True)
class StickyOptions:
    output: Path = Path("runs/ibl_stickyq_calib")
    episodes: int = 15
    trials_per_episode: int = 400
    seed: int = 1234
    learning_rate: float = 0.05
    weight_decay: float = 0.0
    temperature: float = 1.0
    sample_actions: bool = True


@dataclass(slots=True)
class PPOOptions:
    output: Path = Path("runs/rdm_ppo_calib")
    total_timesteps: int = 60_000
    trials_per_episode: int = 600
    eval_episodes: int = 1
    seed: int = 1234
    per_step_cost: float = 0.02
    evidence_gain: float = 0.05
    momentary_sigma: float = 1.0
    include_cumulative_evidence: bool = True
    collapsing_bound: bool = True
    min_bound_steps: int = 20
    bound_threshold: float = 3.0
    # Confidence-based reward parameters
    use_confidence_reward: bool = False
    confidence_bonus_weight: float = 1.0
    base_time_cost: float = 0.0001
    time_cost_growth: float = 0.01
    # PPO hyperparameters
    learning_rate: float = 5e-5
    n_steps: int = 512
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5


@dataclass(slots=True)
class DDMOptions:
    output: Path = Path("runs/rdm_ddm_baseline")
    trials_per_episode: int = 600
    episodes: int = 1
    seed: int = 42
    drift_gain: float = 0.1
    noise: float = 3.0
    bound: float = 12.0
    non_decision_ms: float = 100.0
    per_step_cost: float = 0.02


@dataclass(slots=True)
class Args:
    mode: Literal["sticky", "ppo", "ddm", "all"] = "all"
    sticky: StickyOptions = field(default_factory=StickyOptions)
    ppo: PPOOptions = field(default_factory=PPOOptions)
    ddm: DDMOptions = field(default_factory=DDMOptions)
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
    hyper = StickyGLMHyperParams(
        learning_rate=opts.learning_rate,
        weight_decay=opts.weight_decay,
        temperature=opts.temperature,
        sample_actions=opts.sample_actions,
    )
    cfg = StickyGLMTrainingConfig(
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
        use_confidence_reward=opts.use_confidence_reward,
        confidence_bonus_weight=opts.confidence_bonus_weight,
        base_time_cost=opts.base_time_cost,
        time_cost_growth=opts.time_cost_growth,
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


def run_ddm(opts: DDMOptions, verbose: bool = True) -> Path:
    cfg = DDMConfig(
        trials_per_episode=opts.trials_per_episode,
        episodes=opts.episodes,
        seed=opts.seed,
        drift_gain=opts.drift_gain,
        noise=opts.noise,
        bound=opts.bound,
        non_decision_ms=opts.non_decision_ms,
        per_step_cost=opts.per_step_cost,
        output_dir=opts.output,
    )
    if verbose:
        _print_heading("Running DDM baseline")
        print(json.dumps({
            "trials_per_episode": cfg.trials_per_episode,
            "bound": cfg.bound,
            "drift_gain": cfg.drift_gain,
        }, indent=2))
    runner = DDMBaseline(cfg)
    metrics = runner.run()
    if verbose:
        _print_heading("DDM metrics")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"Artifacts: {cfg.output_paths()['root']}")
    return cfg.output_paths()["root"]


def main() -> None:
    args = tyro.cli(Args)
    if args.mode in {"sticky", "all"}:
        run_sticky(args.sticky, verbose=args.verbose)
    if args.mode in {"ppo", "all"}:
        run_ppo(args.ppo, verbose=args.verbose)
    if args.mode in {"ddm", "all"}:
        run_ddm(args.ddm, verbose=args.verbose)


if __name__ == "__main__":
    main()
