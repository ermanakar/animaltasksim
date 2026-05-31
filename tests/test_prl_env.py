from __future__ import annotations

import json

import pytest

from envs.ibl_2afc import ACTION_LEFT, ACTION_NO_OP, ACTION_RIGHT, AgentMetadata
from envs.prl_reversal import ContingencyBlock, PRLConfig, ProbabilisticReversalLearningEnv
from envs.utils_timing import PhaseTiming
from eval.schema_validator import validate_file


@pytest.fixture()
def prl_short_config(tmp_path):
    return PRLConfig(
        trials_per_episode=2,
        blocks=(
            ContingencyBlock(p_left_reward=1.0, p_right_reward=0.0, length=1),
            ContingencyBlock(p_left_reward=0.0, p_right_reward=1.0, length=1),
        ),
        phase_schedule=(
            PhaseTiming("iti", 1),
            PhaseTiming("stimulus", 1),
            PhaseTiming("response", 2),
            PhaseTiming("outcome", 1),
        ),
        step_ms=50,
        include_phase_onehot=True,
        include_timing=True,
        expose_prior=False,
        log_path=tmp_path / "prl_trials.ndjson",
        agent=AgentMetadata(name="tester-prl", version="0.0.1"),
        seed=42,
    )


def test_prl_reset_matches_observation_space(prl_short_config):
    env = ProbabilisticReversalLearningEnv(prl_short_config)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["phase"] == "iti"
    assert info["block_index"] == 0
    assert info["reversal"] is False
    assert obs["contrast"] == 0.0
    assert "block_prior" not in obs
    env.close()


def test_prl_reversal_contingency_and_logging(prl_short_config):
    env = ProbabilisticReversalLearningEnv(prl_short_config)
    log_path = prl_short_config.log_path
    assert log_path is not None
    obs, info = env.reset()

    # Trial 1: Block 0 (Left reward = 1.0, Right reward = 0.0)
    # Step ITI
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    assert reward == 0.0
    # Step Stimulus
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    # Step Response (Choose Left -> which should trigger 1.0 reward on Block 0)
    obs, reward, terminated, truncated, info = env.step(ACTION_LEFT)
    # Step Outcome
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)

    assert pytest.approx(reward) == 1.0
    assert not terminated

    # Trial 2: Block 1 (Left reward = 0.0, Right reward = 1.0)
    # Step ITI
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    # Step Stimulus
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    # Step Response (Choose Right -> which should trigger 1.0 reward on Block 1)
    obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
    # Step Outcome
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)

    assert pytest.approx(reward) == 1.0
    assert terminated
    env.close()

    validate_file(log_path)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    record_1 = json.loads(lines[0])
    assert record_1["task"] == "prl"
    assert record_1["block_index"] == 0
    assert record_1["action"] == "left"
    assert record_1["reward"] == 1.0
    assert record_1["reversal"] is False
    assert record_1["contingency"] == {"left": 1.0, "right": 0.0}

    record_2 = json.loads(lines[1])
    assert record_2["block_index"] == 1
    assert record_2["action"] == "right"
    assert record_2["reward"] == 1.0
    assert record_2["reversal"] is True
    assert record_2["contingency"] == {"left": 0.0, "right": 1.0}


def test_prl_rejects_non_reversing_blocks() -> None:
    with pytest.raises(ValueError, match="reverse"):
        PRLConfig(
            blocks=(
                ContingencyBlock(p_left_reward=0.8, p_right_reward=0.2, length=20),
                ContingencyBlock(p_left_reward=0.7, p_right_reward=0.3, length=20),
            )
        )


def test_prl_rejects_non_reversing_wraparound() -> None:
    with pytest.raises(ValueError, match="reverse"):
        PRLConfig(
            blocks=(
                ContingencyBlock(p_left_reward=0.8, p_right_reward=0.2, length=20),
                ContingencyBlock(p_left_reward=0.2, p_right_reward=0.8, length=20),
                ContingencyBlock(p_left_reward=0.8, p_right_reward=0.2, length=20),
            )
        )


def test_prl_rejects_impossible_latency() -> None:
    with pytest.raises(ValueError, match="smaller than response"):
        PRLConfig(min_response_latency_steps=30)
