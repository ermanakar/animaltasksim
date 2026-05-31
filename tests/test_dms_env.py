from __future__ import annotations

import json

import pytest

from envs.dms_match import DMSConfig, DelayedMatchToSampleEnv
from envs.ibl_2afc import ACTION_LEFT, ACTION_NO_OP, ACTION_RIGHT, AgentMetadata
from envs.utils_timing import PhaseTiming
from eval.schema_validator import validate_file


@pytest.fixture()
def dms_short_config(tmp_path):
    return DMSConfig(
        trials_per_episode=1,
        contrast_set=(1.0,),
        phase_schedule=(
            PhaseTiming("iti", 1),
            PhaseTiming("sample", 1),
            PhaseTiming("delay", 1),
            PhaseTiming("test", 1),
            PhaseTiming("response", 1),
            PhaseTiming("outcome", 1),
        ),
        step_ms=10,
        include_phase_onehot=True,
        include_timing=True,
        expose_prior=False,
        log_path=tmp_path / "dms_trials.ndjson",
        agent=AgentMetadata(name="tester-dms", version="0.0.1"),
        seed=101,
    )


def test_dms_reset_matches_observation_space(dms_short_config):
    env = DelayedMatchToSampleEnv(dms_short_config)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["phase"] == "iti"
    assert "sample_stimulus" in info
    assert "match" in info
    assert info["delay_ms"] == 10.0
    env.close()


def test_dms_phases_and_match_logging(dms_short_config):
    env = DelayedMatchToSampleEnv(dms_short_config)
    log_path = dms_short_config.log_path
    assert log_path is not None
    obs, info = env.reset()

    # At reset: ITI phase
    assert obs["contrast"] == 0.0
    is_match = info["match"]
    sample_contrast = info["sample_stimulus"]["contrast"]

    # step() inside ITI advances the environment to the "sample" phase
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    assert obs["contrast"] == sample_contrast

    # step() inside sample advances the environment to the "delay" phase
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    assert obs["contrast"] == 0.0

    # step() inside delay advances the environment to the "test" phase
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    if is_match:
        assert obs["contrast"] == sample_contrast
    else:
        assert obs["contrast"] == -sample_contrast

    # step() inside test advances the environment to the "response" phase
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)

    # We are in response phase; chosen action classifies Match (ACTION_RIGHT) or Non-Match (ACTION_LEFT)
    chosen_action = ACTION_RIGHT if is_match else ACTION_LEFT
    obs, reward, terminated, truncated, info = env.step(chosen_action)

    # Outcome phase: reward is delivered
    obs, reward, terminated, truncated, info = env.step(ACTION_NO_OP)
    assert pytest.approx(reward) == 1.0
    assert terminated
    env.close()

    validate_file(log_path)
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["task"] == "dms"
    assert record["correct"] is True
    assert record["action"] == ("right" if is_match else "left")
    assert record["match"] == is_match
    assert record["delay_ms"] == 10.0
    assert record["sample_stimulus"]["contrast"] == sample_contrast


def test_dms_test_strength_cannot_reveal_match_status(tmp_path) -> None:
    config = DMSConfig(
        contrast_set=(0.25, 1.0),
        log_path=tmp_path / "dms_trials.ndjson",
        seed=101,
    )
    env = DelayedMatchToSampleEnv(config)

    for seed in range(10):
        env.reset(seed=seed)
        assert abs(float(env._test_stimulus["contrast"])) == abs(  # noqa: SLF001
            float(env._sample_stimulus["contrast"])  # noqa: SLF001
        )

    env.close()
