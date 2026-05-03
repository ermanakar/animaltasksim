"""Trainer and rollout entrypoints for the adaptive control agent."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from agents.adaptive_control_config import AdaptiveControlConfig
from agents.adaptive_control_model import AdaptiveControlModel
from agents.hybrid_trainer import HybridDDMTrainer
from animaltasksim.seeding import seed_everything
from envs.ibl_2afc import ACTION_NO_OP, AgentMetadata as IBLAgentMetadata, IBL2AFCConfig, IBL2AFCEnv
from envs.rdm_macaque import ACTION_HOLD, ACTION_LEFT, ACTION_RIGHT, AgentMetadata, RDMConfig, RDMMacaqueEnv
from envs.utils_timing import PhaseTiming


class AdaptiveControlTrainer(HybridDDMTrainer):
    """Adaptive-control trainer reusing the validated hybrid training stack."""

    def __init__(self, config: AdaptiveControlConfig) -> None:
        self.config = config
        self.config.loss_weights.clamp_non_negative()
        self.device = torch.device("cpu")
        seed_everything(self.config.seed)
        self.sessions = self._load_reference_sessions()
        if not self.sessions:
            raise RuntimeError("No reference sessions found for training.")
        self.feature_dim = self.sessions[0].features.shape[1]
        self.model = AdaptiveControlModel(
            feature_dim=self.feature_dim,
            hidden_size=self.config.hidden_size,
            device=self.device,
            drift_scale=self.config.drift_scale,
            history_bias_scale=self.config.history_bias_scale,
            history_drift_scale=self.config.history_drift_scale,
            control_state_enabled=self.config.control_state_enabled,
            persistence_enabled=self.config.persistence_enabled,
            exploration_enabled=self.config.exploration_enabled,
            persistence_learning_rate=self.config.persistence_learning_rate,
            switch_learning_rate=self.config.switch_learning_rate,
            reward_learning_rate=self.config.reward_learning_rate,
            control_state_decay=self.config.control_state_decay,
            control_state_scale=self.config.control_state_scale,
            persistence_bias_scale=self.config.persistence_bias_scale,
            exploration_bias_scale=self.config.exploration_bias_scale,
            control_residual_limit=self.config.control_residual_limit,
            control_pressure_limit=self.config.control_pressure_limit,
            control_uncertainty_power=self.config.control_uncertainty_power,
        )
        self.model.to(self.device)
        if self.config.freeze_history_scales:
            self.model.history_bias_scale.requires_grad_(False)
            self.model.history_drift_scale.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self._history_injection_alpha = 0.0
        self._history_injection_alpha_start = float(self.config.history_injection_alpha_start)
        self._history_injection_alpha_end = float(self.config.history_injection_alpha_end)

    @staticmethod
    def _ibl_phase_schedule(max_commit_steps: int) -> tuple[PhaseTiming, ...]:
        """Build an IBL phase schedule with the trained DDM response window."""
        return (
            PhaseTiming("iti", 10),
            PhaseTiming("stimulus", 10),
            PhaseTiming("response", max_commit_steps),
            PhaseTiming("outcome", 10),
        )

    def rollout(self, paths) -> dict[str, float]:
        """Run episodes in the environment and write schema-valid `.ndjson` logs."""
        if self.config.task == "ibl_2afc":
            ibl_config = IBL2AFCConfig(
                trials_per_episode=self.config.trials_per_episode,
                log_path=paths.log,
                agent=IBLAgentMetadata(name="adaptive_control", version=self.config.agent_version),
                seed=self.config.seed,
                phase_schedule=self._ibl_phase_schedule(self.config.max_commit_steps),
                min_response_latency_steps=self.config.min_commit_steps,
            )
            env = IBL2AFCEnv(ibl_config)
            wait_action = ACTION_NO_OP
            new_trial_phase = "iti"
        else:
            rdm_config = RDMConfig(
                trials_per_episode=self.config.trials_per_episode,
                log_path=paths.log,
                agent=AgentMetadata(name="adaptive_control", version=self.config.agent_version),
                seed=self.config.seed,
                per_step_cost=0.01,
                evidence_gain=0.05,
                momentary_sigma=1.0,
                collapsing_bound=False,
                min_bound_steps=self.config.min_commit_steps,
                response_duration_override=self.config.max_commit_steps,
            )
            env = RDMMacaqueEnv(rdm_config)
            wait_action = ACTION_HOLD
            new_trial_phase = "fixation"

        step_ms = env.config.step_ms
        response_phase = next(p for p in env._phase_schedule if p.name == "response")  # noqa: SLF001
        effective_max_commit = min(self.config.max_commit_steps, response_phase.duration_steps)
        metrics: dict[str, list[float]] = {"cumulative_reward": [], "mean_rt_ms": []}
        for episode in range(self.config.episodes):
            _, info = env.reset(seed=self.config.seed + episode)
            h, c = self.model.init_state()
            plastic_state, eligibility_trace, prev_value_prediction, prev_history_gate = self.model.init_plastic_state()
            cumulative_reward = 0.0
            planned_action = wait_action
            commit_step_target = self.config.min_commit_steps
            rt_tracker: list[float] = []
            prev_action_val = 0.0
            prev_reward = 0.0
            prev_correct = 0.0
            current_stimulus = self._get_stimulus(env)
            while True:
                phase = info["phase"]
                if phase == "response":
                    if info["phase_step"] == 0:
                        current_stimulus = self._get_stimulus(env)
                        trial_idx_val = info.get("trial_index", 0)
                        trial_idx = int(trial_idx_val) if isinstance(trial_idx_val, (int, float)) else 0
                        trial_norm = float(trial_idx) / max(self.config.trials_per_episode, 1)
                        features = self._features_from_trial(
                            current_stimulus,
                            prev_action_val,
                            prev_reward,
                            prev_correct,
                            trial_norm,
                        )
                        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
                        plastic_state, eligibility_trace, _ = self.model.update_plastic_history(
                            plastic_state=plastic_state,
                            eligibility_trace=eligibility_trace,
                            prev_action=x[:, 3],
                            prev_reward=x[:, 4],
                            prev_value_prediction=prev_value_prediction,
                            prev_history_gate=prev_history_gate,
                        )
                        out, (h, c) = self.model(x, (h, c), plastic_state=plastic_state)
                        prev_value_prediction = out["critic_value"].reshape(-1, 1)
                        prev_history_gate = torch.clamp(1.0 - torch.abs(x[:, 0:1]), min=0.0, max=1.0)

                        stimulus = x[0, 0].item()
                        drift_gain = out["drift_gain"].item()
                        bound = out["bound"].item()
                        noise = out["noise"].item()
                        bias = out["bias"].item()
                        stay_tendency = out["stay_tendency"].item()
                        stay_tendency = self._apply_rollout_history_policy(stay_tendency, prev_reward)
                        non_decision_ms = out["non_decision_ms"].item()

                        prev_direction = 1.0 if prev_action_val == 1 else (-1.0 if prev_action_val == -1 else 0.0)
                        hb_scale = self.model.effective_history_bias_scale.item()
                        hd_scale = self.model.history_drift_scale.item()
                        stay_shift = stay_tendency * hb_scale * bound
                        combined_start = bias + stay_shift * prev_direction
                        history_drift = stay_tendency * hd_scale * prev_direction
                        confidence = min(abs(stimulus), 1.0)
                        gated_history_drift = history_drift * (1.0 - confidence)
                        drift = drift_gain * stimulus + gated_history_drift
                        dt = self.config.step_ms / 1000.0

                        if np.random.random() < self.config.lapse_rate:
                            planned_action = int(np.random.choice([ACTION_LEFT, ACTION_RIGHT]))
                            ddm_steps = int(np.random.randint(self.config.min_commit_steps, effective_max_commit + 1))
                        else:
                            planned_action, ddm_steps = self._simulate_ddm(
                                drift=drift,
                                bound=bound,
                                noise=noise,
                                bias=combined_start,
                                dt=dt,
                                max_steps=effective_max_commit,
                            )

                        ddm_time_ms = ddm_steps * self.config.step_ms
                        total_rt_ms = non_decision_ms + ddm_time_ms
                        total_rt_ms = np.clip(
                            total_rt_ms,
                            self.config.step_ms * self.config.min_commit_steps,
                            self.config.step_ms * effective_max_commit,
                        )
                        commit_step_target = int(total_rt_ms / step_ms)
                        commit_step_target = int(np.clip(commit_step_target, self.config.min_commit_steps, effective_max_commit))
                    phase_step_val = info.get("phase_step", 0)
                    current_step = int(phase_step_val) if isinstance(phase_step_val, (int, float)) else 0
                    action = planned_action if current_step + 1 >= commit_step_target else wait_action
                else:
                    action = wait_action

                _, reward, terminated, _, info = env.step(action)
                cumulative_reward += float(reward)
                if info["phase"] == "outcome" and info.get("phase_step", 0) == 1:
                    actual_rt_ms = float((env._rt_steps or 0) * step_ms)  # noqa: SLF001
                    if actual_rt_ms > 0:
                        rt_tracker.append(actual_rt_ms)
                    if planned_action == ACTION_RIGHT:
                        prev_action_val = 1.0
                    elif planned_action == ACTION_LEFT:
                        prev_action_val = -1.0
                    else:
                        prev_action_val = 0.0
                    if self.config.task == "ibl_2afc" and abs(current_stimulus) < 1e-6:
                        block_prior = float(info.get("block_prior", 0.5))
                        expected_right = block_prior > 0.5
                    else:
                        expected_right = current_stimulus >= 0.0
                    prev_correct = 1.0 if (planned_action == ACTION_RIGHT) == expected_right else 0.0
                    prev_reward = float(reward)
                if info["phase"] == new_trial_phase and info.get("phase_step", 0) == 0:
                    pass
                if terminated:
                    metrics["cumulative_reward"].append(cumulative_reward)
                    metrics["mean_rt_ms"].append(float(np.mean(rt_tracker) if rt_tracker else 0.0))
                    break
        env.close()
        return {
            "mean_reward": float(np.mean(metrics["cumulative_reward"])) if metrics["cumulative_reward"] else 0.0,
            "mean_rt_ms": float(np.mean(metrics["mean_rt_ms"])) if metrics["mean_rt_ms"] else 0.0,
        }


def train_adaptive_control(config: AdaptiveControlConfig) -> dict[str, Any]:
    """High-level training entry point for the adaptive control family."""
    trainer = AdaptiveControlTrainer(config)
    trainer.config.max_commit_steps = max(trainer.config.max_commit_steps, 180)
    paths = config.output_paths()
    training_metrics = trainer.train()
    rollout_stats = trainer.rollout(paths)
    trainer.save(paths, training_metrics, rollout_stats)
    return {
        "training_metrics": training_metrics,
        "rollout_stats": rollout_stats,
        "paths": {
            "log": str(paths.log),
            "config": str(paths.config),
            "metrics": str(paths.metrics),
            "model": str(paths.model),
        },
    }


__all__ = ["AdaptiveControlTrainer", "train_adaptive_control"]
