from __future__ import annotations

import json

import numpy as np
import pytest

from envs.ibl_2afc import (
    ACTION_LEFT,
    ACTION_NO_OP,
    ACTION_RIGHT,
    ACTION_NAMES,
    AgentMetadata,
    BlockConfig,
    IBL2AFCConfig,
    IBL2AFCEnv,
)
from envs.utils_timing import PhaseTiming
from envs.rdm_macaque import (
    ACTION_HOLD,
    ACTION_LEFT as RDM_ACTION_LEFT,
    ACTION_NAMES as RDM_ACTION_NAMES,
    ACTION_RIGHT as RDM_ACTION_RIGHT,
    AgentMetadata as RDMAgentMetadata,
    RDMConfig,
    RDMMacaqueEnv,
)
from eval.schema_validator import validate_file


@pytest.fixture()
def short_config(tmp_path):
    return IBL2AFCConfig(
        trials_per_episode=1,
        contrast_set=(0.25,),
        block_sequence=(BlockConfig(p_right=1.0, length=1),),
        phase_schedule=(
            PhaseTiming("iti", 1),
            PhaseTiming("stimulus", 1),
            PhaseTiming("response", 2),
            PhaseTiming("outcome", 1),
        ),
        step_ms=50,
        include_phase_onehot=True,
        include_timing=True,
        expose_prior=True,
        log_path=tmp_path / "trials.ndjson",
        agent=AgentMetadata(name="tester", version="0.0.1"),
        seed=7,
    )


def _collect_episode(env: IBL2AFCEnv) -> list[tuple[dict, float, bool]]:
    """Run an episode while recording observations, rewards, and done flags."""

    trajectory: list[tuple[dict, float, bool]] = []
    obs, _info = env.reset()
    terminated = False
    while not terminated:
        phase_hot = obs.get("phase_onehot")
        phase_index = int(np.argmax(phase_hot)) if phase_hot is not None else None
        phase_name = env._phase_names[phase_index] if phase_index is not None else ""
        if phase_name == "response":
            action = ACTION_RIGHT
        else:
            action = ACTION_NO_OP
        obs, reward, terminated, _truncated, info = env.step(action)
        trajectory.append((obs, reward, terminated))
    return trajectory


def test_reset_matches_observation_space(short_config):
    env = IBL2AFCEnv(short_config)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["phase"] == "iti"
    env.close()


def test_trial_reward_and_logging(short_config, tmp_path):
    env = IBL2AFCEnv(short_config)
    log_path = short_config.log_path
    assert log_path is not None
    obs, info = env.reset()

    # ITI step
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    assert reward == 0.0
    assert not terminated
    # Stimulus step
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    assert reward == 0.0
    # Response step (respond right)
    obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
    assert reward == 0.0
    assert not terminated
    # Outcome step should deliver reward
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    assert pytest.approx(reward) == 1.0
    assert terminated
    assert not truncated
    env.close()

    validate_file(log_path)
    payloads = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(payloads) == 1
    record = payloads[0]
    assert record["correct"] is True
    assert record["action"] == "right"
    assert record["reward"] == 1.0
    assert record["phase_times"]["response_ms"] > 0
    assert record["agent"]["name"] == "tester"


def test_timeout_logs_no_op(short_config, tmp_path):
    config = short_config
    config = IBL2AFCConfig(
        trials_per_episode=1,
        contrast_set=(0.25,),
        block_sequence=(BlockConfig(p_right=0.0, length=1),),
        phase_schedule=short_config.phase_schedule,
        step_ms=short_config.step_ms,
        include_phase_onehot=False,
        include_timing=False,
        expose_prior=False,
        log_path=tmp_path / "timeout.ndjson",
        agent=AgentMetadata(name="tester", version="0.0.1"),
        seed=11,
    )
    env = IBL2AFCEnv(config)
    obs, info = env.reset()
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    env.close()

    log_path = config.log_path
    assert log_path is not None
    validate_file(log_path)
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["action"] == "no-op"
    assert record["correct"] is False
    assert record["reward"] == 0.0


@pytest.fixture()
def rdm_config(tmp_path):
    return RDMConfig(
        trials_per_episode=1,
        coherence_set=(0.5,),
        phase_schedule=(
            PhaseTiming("fixation", 1),
            PhaseTiming("stimulus", 1),
            PhaseTiming("response", 3),
            PhaseTiming("outcome", 1),
        ),
        step_ms=40,
        include_phase_onehot=True,
        include_timing=True,
        per_step_cost=0.05,
        collapsing_bound=True,
        min_bound_steps=1,
        log_path=tmp_path / "rdm.ndjson",
        agent=RDMAgentMetadata(name="ppo", version="0.0.1"),
        seed=5,
    )


def test_rdm_env_response_and_logging(rdm_config):
    env = RDMMacaqueEnv(rdm_config)
    log_path = rdm_config.log_path
    assert log_path is not None
    obs, info = env.reset()
    assert "coherence" in obs
    assert "cumulative_evidence" in obs

    # fixation
    obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
    assert reward == 0.0
    # stimulus
    obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
    assert reward == 0.0
    # response (match stimulus direction)
    direction = env._stimulus["direction"]
    chosen = RDM_ACTION_RIGHT if direction == "right" or direction == "none" else RDM_ACTION_LEFT
    while info["phase"] == "response" and not terminated:
        obs, reward, terminated, truncated, info = env.step(chosen)
    assert reward <= 0  # cost applied during response
    if not terminated:
        obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
    assert terminated is True
    assert reward < 1.0
    env.close()

    validate_file(log_path)
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["task"] == "rdm"
    assert record["correct"] is True
    assert record["action"] == RDM_ACTION_NAMES[chosen]
    assert record["phase_times"]["response_ms"] > 0


def test_rdm_timeout_logs_hold(rdm_config, tmp_path):
    config = RDMConfig(
        trials_per_episode=1,
        coherence_set=(0.5,),
        phase_schedule=rdm_config.phase_schedule,
        step_ms=rdm_config.step_ms,
        include_phase_onehot=False,
        include_timing=False,
        per_step_cost=0.0,
        collapsing_bound=True,
        min_bound_steps=2,
        log_path=tmp_path / "rdm_timeout.ndjson",
        agent=RDMAgentMetadata(name="ppo", version="0.0.1"),
        seed=9,
    )

    env = RDMMacaqueEnv(config)
    obs, info = env.reset()
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)

    env.close()
    log_path = config.log_path
    assert log_path is not None
    validate_file(log_path)
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["action"] == "hold"
    assert record["correct"] is False
    assert record["reward"] == 0.0
