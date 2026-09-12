"""Regression checks for observable feedback and matched control comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.adaptive_control_config import AdaptiveControlConfig
from agents.adaptive_control_model import AdaptiveControlModel
from agents.adaptive_control_trainer import AdaptiveControlTrainer
from envs.ibl_2afc import ACTION_NO_OP, ACTION_RIGHT, BlockConfig, IBL2AFCConfig, IBL2AFCEnv
from envs.prl_reversal import PRLConfig, ProbabilisticReversalLearningEnv
from envs.utils_timing import PhaseTiming
from eval.schema_validator import validate_file


def test_zero_contrast_reward_uses_sampled_target() -> None:
    env = IBL2AFCEnv(IBL2AFCConfig(
        contrast_set=(0.0,), block_sequence=(BlockConfig(p_right=0.8, length=500),),
    ))
    rewarded = []
    for seed in range(500):
        env.reset(seed=seed)
        assert env._stimulus == {"contrast": 0.0, "side": "none"}
        env._phase_index = 2
        env._process_response(ACTION_RIGHT)
        rewarded.append(env._correct)
    assert 0.7 < np.mean(rewarded) < 0.9
    env.close()


@pytest.mark.parametrize("kind", ["ibl", "prl"])
def test_history_updates_without_a_logger(kind: str) -> None:
    schedule = tuple(PhaseTiming(name, 1) for name in ("iti", "stimulus", "response", "outcome"))
    config_type = IBL2AFCConfig if kind == "ibl" else PRLConfig
    env_type = IBL2AFCEnv if kind == "ibl" else ProbabilisticReversalLearningEnv
    env = env_type(config_type(trials_per_episode=2, phase_schedule=schedule, include_history=True))
    _, info = env.reset(seed=10)
    while info["trial_index"] == 0:
        obs, _, done, _, info = env.step(ACTION_RIGHT if info["phase"] == "response" else ACTION_NO_OP)
        assert not done
    assert obs["prev_action"][ACTION_RIGHT] == 1.0
    assert float(obs["prev_reward"]) != 0.0
    if kind == "prl":
        assert float(obs["prev_correct"]) == float(obs["prev_reward"] > 0.0)
        env._prev_correct = not env._prev_correct
        assert float(env._build_observation()["prev_correct"]) == float(obs["prev_correct"])
    env.close()


def test_zero_residual_matches_disabled_control_with_nonzero_base_history() -> None:
    model = AdaptiveControlModel(feature_dim=7, hidden_size=8, device=torch.device("cpu"))
    with torch.no_grad():
        model.win_history_network[-1].bias.fill_(0.8)
    x = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]])
    enabled, _ = model(x, model.init_state())
    model.control_state_enabled = False
    disabled, _ = model(x, model.init_state())
    assert enabled["base_stay_tendency"].abs().item() > 0.1
    assert torch.equal(enabled["stay_tendency"], disabled["stay_tendency"])


def test_prl_rollout_only_uses_observable_feedback_and_configured_timing(tmp_path: Path) -> None:
    config = AdaptiveControlConfig(task="prl", output_dir=tmp_path, trials_per_episode=30,
                                   episodes=1, step_ms=25, hidden_size=8, max_commit_steps=30)
    trainer = AdaptiveControlTrainer.__new__(AdaptiveControlTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.model = AdaptiveControlModel(feature_dim=7, hidden_size=8, device=trainer.device)
    trainer._history_injection_alpha = 0.0
    inputs: list[torch.Tensor] = []
    hook = trainer.model.register_forward_pre_hook(lambda _model, args: inputs.append(args[0].clone()))
    diagnostics = tmp_path / "diagnostics.ndjson"
    paths = config.output_paths()
    trainer.rollout(paths, diagnostics_path=diagnostics)
    hook.remove()
    validate_file(paths.log)
    records = [json.loads(line) for line in paths.log.read_text().splitlines()]
    traces = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert len(records) == len(inputs) == len(traces) == 30
    for index, record in enumerate(records):
        assert record["phase_times"]["stimulus_ms"] == 250
        assert record["rt_ms"] == traces[index]["commit_step_target"] * 25
        if index:
            assert inputs[index][0, 5].item() == float(records[index - 1]["reward"] > 0.0)


def test_paired_trial_lapses_ignore_prior_ddm_random_consumption(tmp_path: Path) -> None:
    trial_results = []
    for noise_draws in (1, 100):
        config = AdaptiveControlConfig(task="prl", output_dir=tmp_path / str(noise_draws),
                                       trials_per_episode=25, episodes=1, lapse_rate=0.5,
                                       hidden_size=8, max_commit_steps=30)
        trainer = AdaptiveControlTrainer.__new__(AdaptiveControlTrainer)
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.model = AdaptiveControlModel(feature_dim=7, hidden_size=8, device=trainer.device)
        trainer._history_injection_alpha = 0.0

        def simulate(**kwargs: float) -> tuple[int, int]:
            assert not torch.is_grad_enabled()
            np.random.normal(size=noise_draws)
            return ACTION_RIGHT, 7

        trainer._simulate_ddm = simulate
        paths = config.output_paths()
        diagnostic_path = paths.root / "diagnostics.ndjson"
        trainer.rollout(paths, diagnostics_path=diagnostic_path, paired_trial_seed=999)
        rows = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
        trial_results.append([(row["planned_action"], row["ddm_steps"]) for row in rows])
    assert trial_results[0] == trial_results[1]


def test_adaptive_preserves_base_history_readout_semantics() -> None:
    from agents.hybrid_model import HybridDDMModel

    model = AdaptiveControlModel(feature_dim=7, hidden_size=8, device=torch.device("cpu"))
    with torch.no_grad():
        model.win_history_network[-1].bias.fill_(0.4)
        model.lose_history_network[-1].bias.fill_(0.2)
    x = torch.tensor([[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.2]])
    base, _ = HybridDDMModel.forward(model, x, model.init_state(), plastic_state=torch.zeros(1, 2))
    for enabled in [False, True]:
        model.control_state_enabled = enabled
        output, _ = model(x, model.init_state(), plastic_state=torch.ones(1, 2))
        for key in ["win_stay_tendency", "lose_shift_tendency", "lose_stay_tendency"]:
            torch.testing.assert_close(output[key], base[key])
