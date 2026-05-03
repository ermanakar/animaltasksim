from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch")

from agents.adaptive_control_agent import AdaptiveControlConfig, AdaptiveControlModel, train_adaptive_control
from agents.adaptive_control_trainer import AdaptiveControlTrainer
from agents.losses import LossWeights
from eval.schema_validator import validate_file


def test_adaptive_control_training_smoke(tmp_path: Path) -> None:
    config = AdaptiveControlConfig(
        reference_log=Path("data/ibl/reference.ndjson"),
        output_dir=tmp_path / "adaptive_run",
        trials_per_episode=30,
        episodes=1,
        epochs=1,
        hidden_size=16,
        learning_rate=5e-4,
        loss_weights=LossWeights(choice=1.0, rt=0.1, history=0.0),
        max_sessions=1,
        max_trials_per_session=30,
        min_commit_steps=5,
        max_commit_steps=80,
    )

    result = train_adaptive_control(config)

    paths = result["paths"]
    log_path = Path(paths["log"])
    assert log_path.exists()
    validate_file(log_path)

    with log_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    assert lines
    first = json.loads(lines[0])
    assert first["task"] == "ibl_2afc"
    assert first["agent"]["name"] == "adaptive_control"

    metrics_path = Path(paths["metrics"])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "training" in payload
    assert "epoch_evidence_preservation" in payload["training"]
    assert np.isfinite(payload["rollout"]["mean_reward"])


def test_adaptive_control_defaults_match_ibl_calibration() -> None:
    config = AdaptiveControlConfig()

    assert config.drift_scale == 6.0
    assert config.persistence_bias_scale == 1.6
    assert config.control_uncertainty_power == 2.0


def test_low_confidence_failure_increases_retry_pressure() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
    )
    plastic_state, eligibility_trace, _, _ = model.init_plastic_state()
    prev_value_prediction = torch.tensor([[1.0]])
    prev_action = torch.tensor([1.0])
    prev_reward = torch.tensor([0.0])

    low_conf_state, _, _ = model.update_plastic_history(
        plastic_state=plastic_state.clone(),
        eligibility_trace=eligibility_trace.clone(),
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_value_prediction=prev_value_prediction,
        prev_history_gate=torch.tensor([[1.0]]),
    )
    high_conf_state, _, _ = model.update_plastic_history(
        plastic_state=plastic_state.clone(),
        eligibility_trace=eligibility_trace.clone(),
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_value_prediction=prev_value_prediction,
        prev_history_gate=torch.tensor([[0.0]]),
    )

    low_conf_features = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]], dtype=torch.float32)
    high_conf_features = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5]], dtype=torch.float32)
    low_outputs, _ = model(low_conf_features, model.init_state(), plastic_state=low_conf_state)
    high_outputs, _ = model(high_conf_features, model.init_state(), plastic_state=high_conf_state)

    assert low_conf_state[0, 1].item() > high_conf_state[0, 1].item()
    assert low_outputs["plastic_stay_tendency"].item() > high_outputs["plastic_stay_tendency"].item()
    assert low_outputs["stay_tendency"].item() > high_outputs["stay_tendency"].item()


def test_adaptive_control_ibl_phase_schedule_uses_configured_response_window() -> None:
    phases = AdaptiveControlTrainer._ibl_phase_schedule(max_commit_steps=180)
    response = next(phase for phase in phases if phase.name == "response")

    assert response.duration_steps == 180


def test_persistence_ablation_reduces_retry_pressure() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
        persistence_enabled=True,
    )
    with torch.no_grad():
        model.persistence_head[-1].bias.fill_(2.0)
    prev_value_prediction = torch.tensor([[1.0]])
    prev_action = torch.tensor([1.0])
    prev_reward = torch.tensor([0.0])
    low_conf_features = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]], dtype=torch.float32)

    state_on, trace_on, _, _ = model.init_plastic_state()
    state_on, trace_on, _ = model.update_plastic_history(
        plastic_state=state_on,
        eligibility_trace=trace_on,
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_value_prediction=prev_value_prediction,
        prev_history_gate=torch.tensor([[1.0]]),
    )
    outputs_on, _ = model(low_conf_features, model.init_state(), plastic_state=state_on)

    model.persistence_enabled = False
    state_off, trace_off, _, _ = model.init_plastic_state()
    state_off, trace_off, _ = model.update_plastic_history(
        plastic_state=state_off,
        eligibility_trace=trace_off,
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_value_prediction=prev_value_prediction,
        prev_history_gate=torch.tensor([[1.0]]),
    )
    outputs_off, _ = model(low_conf_features, model.init_state(), plastic_state=state_off)

    assert outputs_on["persistence_pressure"].item() > 0.0
    assert outputs_off["persistence_pressure"].item() == pytest.approx(0.0)
    assert outputs_on["stay_tendency"].item() > outputs_off["stay_tendency"].item()


def test_control_state_disable_zeroes_all_adaptive_outputs() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
        control_state_enabled=False,
        persistence_enabled=True,
        exploration_enabled=True,
    )
    plastic_state = torch.tensor([[2.0, -2.0]], dtype=torch.float32)
    eligibility_trace = torch.ones_like(plastic_state)
    updated_state, updated_trace, delta = model.update_plastic_history(
        plastic_state=plastic_state,
        eligibility_trace=eligibility_trace,
        prev_action=torch.tensor([1.0]),
        prev_reward=torch.tensor([0.0]),
        prev_value_prediction=torch.tensor([[1.0]]),
        prev_history_gate=torch.tensor([[1.0]]),
    )

    assert torch.allclose(updated_state, torch.zeros_like(updated_state))
    assert torch.allclose(updated_trace, torch.zeros_like(updated_trace))
    assert delta.item() == pytest.approx(-1.0)

    features = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]],
        dtype=torch.float32,
    )
    outputs, _ = model(features, model.init_state(), plastic_state=plastic_state)

    for key in [
        "plastic_stay_tendency",
        "raw_control_bias",
        "control_gate",
        "retry_pressure",
        "switch_pressure",
        "win_stay_tendency",
        "lose_shift_tendency",
        "lose_stay_tendency",
        "persistence_pressure",
        "exploration_pressure",
        "staleness_signal",
        "raw_control_residual",
        "control_residual",
        "arbitration_adjustment",
    ]:
        assert outputs[key].item() == pytest.approx(0.0), key


def test_failure_valence_updates_control_state_even_when_critic_expects_failure() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
    )
    plastic_state, eligibility_trace, _, _ = model.init_plastic_state()

    low_conf_state, _, delta = model.update_plastic_history(
        plastic_state=plastic_state.clone(),
        eligibility_trace=eligibility_trace.clone(),
        prev_action=torch.tensor([1.0]),
        prev_reward=torch.tensor([0.0]),
        prev_value_prediction=torch.tensor([[0.0]]),
        prev_history_gate=torch.tensor([[1.0]]),
    )

    assert delta.item() == pytest.approx(0.0)
    assert low_conf_state[0, 1].item() > 0.0
    assert low_conf_state[0, 0].item() == pytest.approx(0.0)


def test_success_valence_reinforces_stay_even_when_critic_expects_success() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
    )
    plastic_state, eligibility_trace, _, _ = model.init_plastic_state()

    rewarded_state, _, delta = model.update_plastic_history(
        plastic_state=plastic_state,
        eligibility_trace=eligibility_trace,
        prev_action=torch.tensor([-1.0]),
        prev_reward=torch.tensor([1.0]),
        prev_value_prediction=torch.tensor([[1.0]]),
        prev_history_gate=torch.tensor([[1.0]]),
    )

    assert delta.item() == pytest.approx(0.0)
    assert rewarded_state[0, 0].item() > 0.0
    assert rewarded_state[0, 1].item() == pytest.approx(0.0)


def test_adaptive_control_residual_is_bounded_and_zero_centered() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
        control_residual_limit=0.2,
        control_pressure_limit=0.1,
    )
    weak_features = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]],
        dtype=torch.float32,
    )

    neutral_outputs, _ = model(
        weak_features,
        model.init_state(),
        plastic_state=torch.zeros(1, 2),
    )
    assert neutral_outputs["persistence_pressure"].item() == pytest.approx(0.0)
    assert neutral_outputs["exploration_pressure"].item() == pytest.approx(0.0)
    assert neutral_outputs["control_residual"].item() == pytest.approx(0.0)

    saturated_outputs, _ = model(
        weak_features,
        model.init_state(),
        plastic_state=torch.tensor([[50.0, -50.0]], dtype=torch.float32),
    )
    assert abs(saturated_outputs["raw_control_bias"].item()) <= 0.2 + 1e-6
    assert abs(saturated_outputs["control_residual"].item()) <= 0.2 + 1e-6
    assert abs(saturated_outputs["retry_pressure"].item()) <= 0.1 + 1e-6


def test_high_confidence_failure_increases_switch_pressure() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
    )
    plastic_state, eligibility_trace, _, _ = model.init_plastic_state()
    switched_state, _, _ = model.update_plastic_history(
        plastic_state=plastic_state,
        eligibility_trace=eligibility_trace,
        prev_action=torch.tensor([1.0]),
        prev_reward=torch.tensor([0.0]),
        prev_value_prediction=torch.tensor([[1.0]]),
        prev_history_gate=torch.tensor([[0.0]]),
    )

    weak_features = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]],
        dtype=torch.float32,
    )
    outputs, _ = model(weak_features, model.init_state(), plastic_state=switched_state)

    assert switched_state[0, 0].item() > 0.0
    assert switched_state[0, 1].item() < 0.0
    assert outputs["switch_pressure"].item() > outputs["retry_pressure"].item()
    assert outputs["raw_control_bias"].item() < 0.0
    assert outputs["plastic_stay_tendency"].item() < 0.0


def test_strong_current_evidence_suppresses_adaptive_control_bias() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
    )
    retry_state = torch.tensor([[-1.0, 2.0]], dtype=torch.float32)
    weak_features = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]],
        dtype=torch.float32,
    )
    strong_features = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]],
        dtype=torch.float32,
    )

    weak_outputs, _ = model(weak_features, model.init_state(), plastic_state=retry_state)
    strong_outputs, _ = model(strong_features, model.init_state(), plastic_state=retry_state)

    assert weak_outputs["raw_control_bias"].item() > 0.0
    assert strong_outputs["raw_control_bias"].item() > 0.0
    assert weak_outputs["control_gate"].item() == pytest.approx(1.0)
    assert strong_outputs["control_gate"].item() == pytest.approx(0.0)
    assert weak_outputs["plastic_stay_tendency"].item() > 0.0
    assert strong_outputs["plastic_stay_tendency"].item() == pytest.approx(0.0)
    assert strong_outputs["control_residual"].item() == pytest.approx(0.0)
    assert strong_outputs["retry_pressure"].item() == pytest.approx(0.0)
    assert strong_outputs["switch_pressure"].item() == pytest.approx(0.0)


def test_strong_current_evidence_suppresses_arbitration_residual() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
        control_residual_limit=0.3,
        control_pressure_limit=0.3,
    )
    with torch.no_grad():
        model.arbitration_head.weight.zero_()
        model.arbitration_head.bias.fill_(2.0)

    weak_features = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]],
        dtype=torch.float32,
    )
    strong_features = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]],
        dtype=torch.float32,
    )
    neutral_state = torch.zeros(1, 2)

    weak_outputs, _ = model(weak_features, model.init_state(), plastic_state=neutral_state)
    strong_outputs, _ = model(strong_features, model.init_state(), plastic_state=neutral_state)

    assert weak_outputs["arbitration_adjustment"].item() > 0.0
    assert weak_outputs["control_residual"].item() > 0.0
    assert strong_outputs["arbitration_adjustment"].item() == pytest.approx(0.0)
    assert strong_outputs["control_residual"].item() == pytest.approx(0.0)


def test_control_gate_sharpens_around_ambiguous_evidence() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
        control_uncertainty_power=3.0,
    )
    mid_features = torch.tensor(
        [[0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 0.5]],
        dtype=torch.float32,
    )

    outputs, _ = model(mid_features, model.init_state(), plastic_state=torch.zeros(1, 2))

    assert outputs["control_gate"].item() == pytest.approx(0.75**3)


def test_exploration_ablation_reduces_staleness_driven_switch_pressure() -> None:
    model = AdaptiveControlModel(
        feature_dim=7,
        hidden_size=16,
        device=torch.device("cpu"),
        exploration_enabled=True,
    )
    with torch.no_grad():
        for layer in (model.exploration_head[0], model.exploration_head[2]):
            layer.weight.zero_()
            layer.bias.fill_(2.0)

    stale_state = torch.tensor([[1.5, -0.2]], dtype=torch.float32)
    weak_features = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5]], dtype=torch.float32)
    outputs_on, _ = model(weak_features, model.init_state(), plastic_state=stale_state)

    model.exploration_enabled = False
    outputs_off, _ = model(weak_features, model.init_state(), plastic_state=stale_state)

    assert outputs_on["staleness_signal"].item() > 0.0
    assert outputs_on["exploration_pressure"].item() > 0.0
    assert outputs_off["exploration_pressure"].item() == pytest.approx(0.0)
    assert outputs_on["stay_tendency"].item() < outputs_off["stay_tendency"].item()
