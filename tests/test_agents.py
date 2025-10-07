from __future__ import annotations

import json

from agents.bayes_observer import BayesParams, BayesTrainingConfig, run_bayesian_observer
from agents.ppo_baseline import PPOHyperParams, PPOTrainingConfig, train_ppo
from agents.sticky_q import StickyQHyperParams, StickyQTrainingConfig, train_sticky_q
from eval.schema_validator import validate_file
from scripts.evaluate_agent import EvaluateArgs, main as evaluate_main


def test_sticky_q_training_produces_schema_compliant_logs(tmp_path):
    output_dir = tmp_path / "run"
    config = StickyQTrainingConfig(
        episodes=1,
        trials_per_episode=5,
        seed=123,
        output_dir=output_dir,
        hyperparams=StickyQHyperParams(
            learning_rate=0.5,
            discount=0.9,
            epsilon=0.2,
            epsilon_min=0.05,
            epsilon_decay=0.95,
            stickiness=0.5,
        ),
    )

    paths = config.output_paths()
    metrics = train_sticky_q(config)

    log_path = paths["log"]
    config_path = paths["config"]
    metrics_path = paths["metrics"]

    assert log_path.exists()
    assert config_path.exists()
    assert metrics_path.exists()

    validate_file(log_path)

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["agent"]["name"] == "sticky_q"
    assert metrics["episodes"] == 1
    assert metrics_path.read_text(encoding="utf-8")

    # Evaluate CLI should write metrics summary.
    evaluate_args = EvaluateArgs(run=output_dir)
    evaluate_main(evaluate_args)
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "metrics" in metrics_payload
    assert metrics_payload["metrics"]["history"]["sticky_choice"] >= 0


def test_bayes_observer_generates_logs(tmp_path):
    output_dir = tmp_path / "bayes_run"
    config = BayesTrainingConfig(
        env="ibl_2afc",
        episodes=1,
        trials_per_episode=5,
        seed=456,
        output_dir=output_dir,
        params=BayesParams(sensory_sigma=0.3, lapse_rate=0.05, bias=0.0),
    )

    metrics = run_bayesian_observer(config)
    paths = config.output_paths()
    log_path = paths["log"]
    config_path = paths["config"]

    assert log_path.exists()
    assert config_path.exists()
    validate_file(log_path)
    assert metrics["episodes"] == 1

    evaluate_args = EvaluateArgs(run=output_dir)
    evaluate_main(evaluate_args)
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["metrics"]["psychometric"]


def test_ppo_baseline_generates_logs(tmp_path):
    output_dir = tmp_path / "ppo_run"
    config = PPOTrainingConfig(
        env="rdm",
        total_timesteps=200,
        eval_trials=5,
        eval_episodes=1,
        per_step_cost=0.05,
        seed=789,
        output_dir=output_dir,
        hyperparams=PPOHyperParams(
            learning_rate=3e-4,
            n_steps=16,
            batch_size=16,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        ),
    )

    metrics = train_ppo(config)
    paths = config.output_paths()
    log_path = paths["log"]

    assert log_path.exists()
    validate_file(log_path)
    assert metrics["episodes"] == 1

    evaluate_args = EvaluateArgs(run=output_dir)
    evaluate_main(evaluate_args)
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "chronometric" in payload["metrics"]
