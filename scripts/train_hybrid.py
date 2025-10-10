"""Train the hybrid DDM + LSTM agent on macaque reference data."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import tyro  # type: ignore
except ImportError:  # pragma: no cover - tyro optional
    tyro = None

from agents.hybrid_ddm_lstm import HybridTrainingConfig, LossWeights, train_hybrid


@dataclass(slots=True)
class LossWeightArgs:
    choice: float = 1.0
    rt: float = 1.0
    history: float = 0.0
    drift_supervision: float = 0.0
    wfpt: float = 0.0
    drift_magnitude: float = 0.0


@dataclass(slots=True)
class TrainHybridArgs:
    reference_log: Path = Path("data/macaque/reference.ndjson")
    output_dir: Path = Path("runs/rdm_hybrid")
    agent_version: str = "0.1.0"
    trials_per_episode: int = 400
    episodes: int = 10
    seed: int = 1234
    epochs: int = 5
    hidden_size: int = 64
    learning_rate: float = 1e-3
    step_ms: int = 10
    max_sessions: int | None = None
    max_trials_per_session: int | None = None
    min_commit_steps: int = 5
    max_commit_steps: int = 120
    drift_scale: float = 10.0
    loss_weights: LossWeightArgs = field(default_factory=LossWeightArgs)


def main(args: TrainHybridArgs) -> None:
    weights = LossWeights(**asdict(args.loss_weights))
    config = HybridTrainingConfig(
        reference_log=args.reference_log,
        output_dir=args.output_dir,
        agent_version=args.agent_version,
        trials_per_episode=args.trials_per_episode,
        episodes=args.episodes,
        seed=args.seed,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        loss_weights=weights,
        step_ms=args.step_ms,
        max_sessions=args.max_sessions,
        max_trials_per_session=args.max_trials_per_session,
        min_commit_steps=args.min_commit_steps,
        max_commit_steps=args.max_commit_steps,
        drift_scale=args.drift_scale,
    )
    result = train_hybrid(config)
    print(result)


def _parse_args() -> TrainHybridArgs:
    if tyro is not None:
        return tyro.cli(TrainHybridArgs)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_log", type=Path, default=Path("data/macaque/reference.ndjson"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/rdm_hybrid"))
    parser.add_argument("--agent_version", type=str, default="0.1.0")
    parser.add_argument("--trials_per_episode", type=int, default=400)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--step_ms", type=int, default=10)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--max_trials_per_session", type=int, default=None)
    parser.add_argument("--min_commit_steps", type=int, default=5)
    parser.add_argument("--max_commit_steps", type=int, default=120)
    parser.add_argument("--loss_weights.choice", dest="loss_choice", type=float, default=1.0)
    parser.add_argument("--loss_weights.rt", dest="loss_rt", type=float, default=1.0)
    parser.add_argument("--loss_weights.history", dest="loss_history", type=float, default=0.0)

    ns = parser.parse_args()
    loss_weights = LossWeightArgs(
        choice=ns.loss_choice,
        rt=ns.loss_rt,
        history=ns.loss_history,
    )
    return TrainHybridArgs(
        reference_log=ns.reference_log,
        output_dir=ns.output_dir,
        agent_version=ns.agent_version,
        trials_per_episode=ns.trials_per_episode,
        episodes=ns.episodes,
        seed=ns.seed,
        epochs=ns.epochs,
        hidden_size=ns.hidden_size,
        learning_rate=ns.learning_rate,
        step_ms=ns.step_ms,
        max_sessions=ns.max_sessions,
        max_trials_per_session=ns.max_trials_per_session,
        min_commit_steps=ns.min_commit_steps,
        max_commit_steps=ns.max_commit_steps,
        loss_weights=loss_weights,
    )


if __name__ == "__main__":
    main(_parse_args())
