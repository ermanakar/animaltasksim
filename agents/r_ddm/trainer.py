"""Training utilities for the Recurrent Drift-Diffusion Model (R-DDM)."""

from __future__ import annotations

import json
import copy
import math
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from agents.wfpt_loss import wfpt_log_likelihood
from agents.losses import per_trial_history_loss
from animaltasksim.seeding import seed_everything
from envs.ibl_2afc import (
    ACTION_LEFT,
    ACTION_NO_OP,
    ACTION_RIGHT,
    AgentMetadata as IBLAgentMetadata,
    IBL2AFCConfig,
    IBL2AFCEnv,
)
from envs.rdm_macaque import (
    ACTION_LEFT as RDM_ACTION_LEFT,
    ACTION_RIGHT as RDM_ACTION_RIGHT,
    ACTION_HOLD as RDM_ACTION_HOLD,
    AgentMetadata as RDMAgentMetadata,
    RDMConfig,
    RDMMacaqueEnv,
)

from .config import RDDMConfig
from .dataset import IBLRDDMDataset, RDMRDDMDataset, rddm_collate_sessions
from .model import RDDMModel, RDDMOutputs


@dataclass(slots=True)
class RDDMTrainingState:
    """Metrics captured at the end of each training epoch."""

    epoch: int
    total_loss: float
    choice_loss: float
    wfpt_loss: float
    history_loss: float
    non_decision_loss: float
    drift_loss: float
    entropy_loss: float
    choice_kl_loss: float
    per_trial_history_loss: float
    win_stay_pred: float
    lose_shift_pred: float
    accuracy: float
    mean_rt_ms: float
    current_entropy_weight: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class RDDMTrainer:
    """Coordinates data loading, optimisation, and evaluation."""

    def __init__(self, config: RDDMConfig):
        self.config = config
        seed_everything(config.seed)

        self.device = torch.device(config.device)
        if config.task == "rdm_macaque":
            self.dataset = RDMRDDMDataset(config.reference_log, max_sessions=config.max_sessions)
        else:
            self.dataset = IBLRDDMDataset(config.reference_log, max_sessions=config.max_sessions)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=rddm_collate_sessions,
        )

        self.model = RDDMModel(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.config.run_dir / "training_metrics.ndjson"

        # For balanced loss weighting
        all_stim = np.concatenate([s.stimulus for s in self.dataset.sessions])
        self._stimulus_bin_edges = torch.tensor(np.percentile(all_stim, np.linspace(0, 100, 11)), dtype=torch.float32)
        self._current_entropy_weight = self.config.entropy_loss_weight

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def _freeze_bias(self):
        for name, param in self.model.named_parameters():
            if "param_head.bias" in name:
                param.requires_grad = False

    def _unfreeze_bias(self):
        for name, param in self.model.named_parameters():
            if "param_head.bias" in name:
                param.requires_grad = True

    def train(self) -> list[RDDMTrainingState]:
        history: list[RDDMTrainingState] = []
        best_accuracy = -float("inf")
        best_weights: dict[str, torch.Tensor] | None = None
        best_state: RDDMTrainingState | None = None

        if self.config.freeze_bias_epochs > 0:
            print("Freezing policy head bias.")
            self._freeze_bias()

        for epoch in range(self.config.epochs):
            if epoch == self.config.freeze_bias_epochs:
                print("Unfreezing policy head bias.")
                self._unfreeze_bias()

            state = self._run_epoch(epoch)
            history.append(state)
            with self.metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(state.to_json()) + "\n")
                fh.flush()
            if state.accuracy > best_accuracy:
                best_accuracy = state.accuracy
                best_state = state
                best_weights = copy.deepcopy(self.model.state_dict())
        self._save_model()
        self._save_config()
        if best_weights is not None:
            torch.save(best_weights, self.config.run_dir / "model_best.pt")
            if best_state is not None:
                with (self.config.run_dir / "training_best.json").open("w", encoding="utf-8") as fh:
                    json.dump(best_state.to_json(), fh, indent=2)
        return history

    def _run_epoch(self, epoch: int) -> RDDMTrainingState:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_choice = torch.tensor(0.0, device=self.device)
        total_wfpt = torch.tensor(0.0, device=self.device)
        total_history = torch.tensor(0.0, device=self.device)
        total_nd = torch.tensor(0.0, device=self.device)
        total_drift = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        total_choice_kl = torch.tensor(0.0, device=self.device)
        total_per_trial_history = torch.tensor(0.0, device=self.device)
        total_weight = 0.0

        total_correct = torch.tensor(0.0, device=self.device)
        total_trials = torch.tensor(0.0, device=self.device)
        rt_sum = torch.tensor(0.0, device=self.device)

        win_stay_values: list[float] = []
        lose_shift_values: list[float] = []

        for batch in self.dataloader:
            mask = batch["mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            features = self._build_features(batch).to(self.device)
            outputs = self.model(features, lengths)

            loss_dict, stats = self._compute_losses(batch, outputs, mask, epoch)

            loss = loss_dict["total"]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            weight = mask.sum().item()
            total_weight += weight
            total_loss += loss.detach() * weight
            total_choice += loss_dict["choice"].detach() * weight
            total_wfpt += loss_dict["wfpt"].detach() * weight
            total_history += loss_dict["history"].detach() * weight
            total_nd += loss_dict["non_decision"].detach() * weight
            total_drift += loss_dict["drift"].detach() * weight
            total_entropy += loss_dict["entropy"].detach() * weight
            total_choice_kl += loss_dict["choice_kl"].detach() * weight
            total_per_trial_history += loss_dict["per_trial_history"].detach() * weight

            total_correct += stats["correct"]
            total_trials += stats["count"]
            rt_sum += stats["rt_sum"]

            if stats["win_stay"] is not None:
                win_stay_values.append(float(stats["win_stay"]))
            if stats["lose_shift"] is not None:
                lose_shift_values.append(float(stats["lose_shift"]))

        denom = max(total_weight, 1.0)
        avg_entropy = float(total_entropy.item() / denom)

        # Update entropy weight
        if self.config.entropy_target > 0 and self.config.entropy_weight_lr > 0:
            if avg_entropy < 0.05:
                self._current_entropy_weight = min(1.0, self._current_entropy_weight + 0.1)
            else:
                entropy_error = avg_entropy - self.config.entropy_target
                self._current_entropy_weight = self._current_entropy_weight - self.config.entropy_weight_lr * entropy_error
                self._current_entropy_weight = max(0.0, min(1.0, self._current_entropy_weight))

        state = RDDMTrainingState(
            epoch=epoch,
            total_loss=float(total_loss.item() / denom),
            choice_loss=float(total_choice.item() / denom),
            wfpt_loss=float(total_wfpt.item() / denom),
            history_loss=float(total_history.item() / denom),
            non_decision_loss=float(total_nd.item() / denom),
            drift_loss=float(total_drift.item() / denom),
            entropy_loss=float(total_entropy.item() / denom),
            choice_kl_loss=float(total_choice_kl.item() / denom),
            per_trial_history_loss=float(total_per_trial_history.item() / denom),
            win_stay_pred=float(sum(win_stay_values) / max(len(win_stay_values), 1)),
            lose_shift_pred=float(sum(lose_shift_values) / max(len(lose_shift_values), 1)),
            accuracy=float((total_correct / total_trials.clamp(min=1.0)).item()),
            mean_rt_ms=float((rt_sum / total_trials.clamp(min=1.0)).item() * 1000.0),
            current_entropy_weight=self._current_entropy_weight,
        )
        return state

    # ------------------------------------------------------------------ #
    def _build_features(self, batch: Dict[str, Tensor]) -> Tensor:
        prev_action = batch["prev_action"].float()
        prev_left = torch.where(prev_action == 0, 1.0, 0.0)
        prev_right = torch.where(prev_action == 1, 1.0, 0.0)
        prev_unknown = (prev_action < 0).float()
        prev_left = torch.where(prev_unknown > 0, 0.5, prev_left)
        prev_right = torch.where(prev_unknown > 0, 0.5, prev_right)

        stim = batch["stimulus"]
        block_prior = batch["block_prior"]
        features = torch.stack(
            [
                stim,
                stim.abs(),
                block_prior * self.config.prior_feature_scale,
                (block_prior - 0.5) * self.config.prior_feature_scale,
                prev_left * self.config.history_feature_scale,
                prev_right * self.config.history_feature_scale,
                (batch["prev_reward"] - 0.5) * self.config.history_feature_scale,
                (batch["prev_correct"] - 0.5) * self.config.history_feature_scale,
            ],
            dim=-1,
        )
        return features

    def _compute_losses(
        self,
        batch: Dict[str, Tensor],
        outputs: RDDMOutputs,
        mask: Tensor,
        epoch: int,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        actions = batch["actions"].to(self.device)
        rt = batch["rt_seconds"].to(self.device)
        prev_action = batch["prev_action"].to(self.device)
        prev_reward = batch["prev_reward"].to(self.device)

        choice_prob = outputs.choice_prob
        choice_prob = torch.clamp(choice_prob, 1e-4, 1.0 - 1e-4)

        mask_float = mask.float()
        action_float = actions.float()
        choice_loss = -(
            action_float * torch.log(choice_prob) + (1.0 - action_float) * torch.log(1.0 - choice_prob)
        )

        # Balanced loss weighting
        stim = batch["stimulus"].to(self.device)[mask]
        if stim.count_nonzero() > 0:
            target_side = (stim > 0).long()
            nonzero = stim.ne(0)
            num_bins = self._stimulus_bin_edges.numel() - 1
            bins = torch.bucketize(stim[nonzero], self._stimulus_bin_edges.to(stim.device)).clamp(0, num_bins - 1)
            strata = bins * 2 + target_side[nonzero]
            counts = torch.bincount(strata, minlength=num_bins * 2).float().clamp(min=1.0)
            weights = (1.0 / counts)[strata]
            weights = torch.clamp(weights, max=5.0)
            weights = weights / weights.mean().clamp(min=1e-6)
            choice_w = torch.ones_like(stim)
            choice_w[nonzero] = weights
            choice_loss = (choice_w * choice_loss[mask]).sum() / choice_w.sum().clamp(min=1.0)
        else:
            choice_loss = (choice_loss * mask_float).sum() / mask_float.sum().clamp(min=1.0)

        entropy = -(choice_prob * torch.log(choice_prob) + (1.0 - choice_prob) * torch.log(1.0 - choice_prob))
        entropy_loss = (entropy * mask_float).sum() / mask_float.sum().clamp(min=1.0)

        stimulus = batch["stimulus"].to(self.device)

        # Assert that the sign of the stimulus matches the sign of the action
        stim = batch["stimulus"][mask]
        nonzero = stim.ne(0)
        label_side = (stim[nonzero] > 0).float()
        if len(label_side) > 0:
            p_right = label_side.mean().item()
            assert 0.4 <= p_right <= 0.6, f"Label sign imbalance in batch: P(right)={p_right:.3f}"

        target_prob = torch.sigmoid(
            self.config.choice_kl_target_slope * (stimulus / max(self.config.stimulus_scale, 1e-6))
        )
        target_prob = torch.clamp(target_prob, 1e-4, 1.0 - 1e-4)
        kl_term = target_prob * (torch.log(target_prob) - torch.log(choice_prob)) + (
            (1.0 - target_prob) * (torch.log(1.0 - target_prob) - torch.log(1.0 - choice_prob))
        )
        choice_kl_loss = (kl_term * mask_float).sum() / mask_float.sum().clamp(min=1.0)

        valid_mask = mask & (rt > 0.0)
        if valid_mask.any():
            ll = wfpt_log_likelihood(
                choice=actions[valid_mask].float(),
                rt=rt[valid_mask],
                drift=outputs.drift[valid_mask],
                bound=outputs.boundary[valid_mask],
                bias=outputs.bias[valid_mask],
                noise=outputs.noise[valid_mask],
                non_decision=outputs.non_decision[valid_mask],
            )
            wfpt_loss_val = -ll.mean()
        else:
            wfpt_loss_val = torch.zeros(1, device=self.device)[0]

        history_loss_val, hist_stats = self._history_regulariser(
            choice_prob=choice_prob,
            prev_action=prev_action,
            prev_reward=prev_reward,
            mask=mask,
        )

        choice_multiplier = (
            self.config.schedule.choice_warmup_weight
            if epoch < self.config.schedule.warmup_epochs
            else 1.0
        )
        weight_choice = self.config.choice_loss_weight * choice_multiplier
        weight_wfpt = self.config.wfpt_loss_weight
        if epoch < self.config.schedule.enable_wfpt_epoch:
            weight_wfpt = 0.0
        elif epoch < self.config.schedule.enable_wfpt_epoch + self.config.schedule.wfpt_ramp_epochs:
            ramp_progress = (epoch - self.config.schedule.enable_wfpt_epoch + 1) / max(self.config.schedule.wfpt_ramp_epochs, 1)
            weight_wfpt *= ramp_progress
        logging.info(f"wfpt_weight={weight_wfpt} at epoch {epoch}")

        if epoch >= self.config.schedule.enable_history_epoch:
            ramp_progress = (epoch - self.config.schedule.enable_history_epoch + 1) / max(self.config.history_ramp_epochs, 1)
            ramp_progress = float(max(0.0, min(1.0, ramp_progress)))
            weight_history = self.config.history * ramp_progress
        else:
            ramp_progress = 0.0
            weight_history = 0.0

        # Non-decision regulariser (encourage plausible Ter)
        if valid_mask.any():
            nd_reg = ((outputs.non_decision[valid_mask] - self.config.non_decision_target) ** 2).mean()
        else:
            nd_reg = torch.zeros(1, device=self.device)[0]
        weight_nd = self.config.non_decision_reg_weight
        if mask.any():
            drift_norm = torch.tanh(outputs.drift / self.config.max_drift)
            stim_target = torch.tanh(stimulus / max(self.config.stimulus_scale, 1e-6))
            drift_supervision = ((drift_norm - stim_target) ** 2 * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            drift_supervision = torch.zeros(1, device=self.device)[0]
        weight_drift = self.config.drift_supervision_weight

        weight_entropy = self._current_entropy_weight
        weight_choice_kl = self.config.choice_kl_weight

        # after forward
        choice_prob = outputs.choice_prob  # shape [B] or [B,1]
        p_right_batch = choice_prob.mean()

        side_bias_penalty = (p_right_batch - 0.5).abs() * self.config.side_bias

        # Per-trial history loss (stronger per-trial gradient signal)
        pt_hist_loss_val = per_trial_history_loss(
            choice_prob=outputs.choice_prob,
            prev_action=prev_action,
            prev_reward=prev_reward,
            target_win_stay=self.config.target_win_stay,
            target_lose_shift=self.config.target_lose_shift,
            mask=mask,
        )
        weight_pt_history = self.config.per_trial_history_weight * ramp_progress

        total_loss = (
            weight_choice * choice_loss
            + weight_wfpt * wfpt_loss_val
            + weight_history * history_loss_val
            + weight_pt_history * pt_hist_loss_val
            + weight_nd * nd_reg
            + weight_drift * drift_supervision
            + weight_entropy * entropy_loss
            + weight_choice_kl * choice_kl_loss
            + side_bias_penalty
        )

        stats = {
            "correct": ((choice_prob > 0.5).float() == action_float)[mask].float().sum().detach(),
            "count": mask_float.sum().detach(),
            "rt_sum": (rt * mask_float).sum().detach(),
            "win_stay": hist_stats["win_stay"],
            "lose_shift": hist_stats["lose_shift"],
        }

        return (
            {
                "total": total_loss,
                "choice": choice_loss,
                "wfpt": wfpt_loss_val,
                "history": history_loss_val,
                "per_trial_history": pt_hist_loss_val,
                "non_decision": nd_reg,
                "drift": drift_supervision,
                "entropy": entropy_loss,
                "choice_kl": choice_kl_loss,
            },
            stats,
        )

    def _history_regulariser(
        self,
        choice_prob: Tensor,
        prev_action: Tensor,
        prev_reward: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, dict[str, float | None]]:
        stay_prob = torch.where(prev_action == 1, choice_prob, 1.0 - choice_prob)
        switch_prob = torch.where(prev_action == 1, 1.0 - choice_prob, choice_prob)

        win_mask = (prev_reward > 0.5) & (prev_action >= 0) & mask
        lose_mask = (prev_reward <= 0.5) & (prev_action >= 0) & mask

        def _masked_mean(values: Tensor, mask_tensor: Tensor) -> tuple[Tensor, float | None]:
            weight = mask_tensor.float().sum()
            if weight.item() == 0.0:
                return torch.zeros(1, device=values.device)[0], None
            value = (values * mask_tensor.float()).sum() / weight
            return value, float(value.detach().item())

        win_mean, win_scalar = _masked_mean(stay_prob, win_mask)
        lose_mean, lose_scalar = _masked_mean(switch_prob, lose_mask)

        loss = (win_mean - self.config.target_win_stay) ** 2 + (lose_mean - self.config.target_lose_shift) ** 2
        return loss, {"win_stay": win_scalar, "lose_shift": lose_scalar}

    # ------------------------------------------------------------------ #
    # Persistence & rollout
    # ------------------------------------------------------------------ #
    def _save_model(self) -> None:
        torch.save(self.model.state_dict(), self.config.run_dir / "model.pt")

    def _save_config(self) -> None:
        with (self.config.run_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(self.config), fh, indent=2, default=str)

    def rollout(
        self,
        n_trials: int | None = None,
        seed: int | None = None,
        *,
        run_name: str = "r_ddm",
        checkpoint_path: Path | None = None,
        suffix: str = "",
        stochastic: bool = True,
        balanced: bool = False,
    ) -> Path:
        """Generate schema-compliant trials by interacting with the environment."""
        self.model.eval()
        total_trials = n_trials or self.config.rollout_trials

        log_path = self.config.run_dir / f"trials{suffix}.ndjson"
        if log_path.exists():
            log_path.unlink()

        original_state: dict[str, torch.Tensor] | None = None
        if checkpoint_path is not None and checkpoint_path.exists():
            original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)

        seed_value = seed or self.config.seed
        if self.config.task == "rdm_macaque":
            result = self._rollout_rdm(total_trials, seed_value, run_name, log_path, stochastic, balanced)
        else:
            result = self._rollout_ibl(total_trials, seed_value, run_name, log_path, stochastic, balanced)

        if original_state is not None:
            self.model.load_state_dict(original_state)
        return result

    def _rollout_ibl(self, total_trials: int, seed_value: int, run_name: str, log_path: Path, stochastic: bool, balanced: bool) -> Path:
        env_config = IBL2AFCConfig(
            trials_per_episode=total_trials,
            include_history=True,
            expose_prior=True,
            log_path=log_path,
            agent=IBLAgentMetadata(name="r_ddm", version="prototype"),
            seed=seed_value,
        )
        env = IBL2AFCEnv(env_config)
        obs, info = env.reset(seed=seed_value)
        current_trial = 0

        # Note: IBL env handles stimulus balance internally; balanced kwarg unused here.

        device = next(self.model.parameters()).device
        hidden: torch.Tensor | None = None

        prev_action = -1
        prev_reward = 0.0
        prev_correct = False

        response_phase = next(p for p in env_config.phase_schedule if p.name == "response")
        response_window = response_phase.duration_steps
        step_ms = env_config.step_ms

        pending_action = ACTION_NO_OP
        action_delay = 0
        trial_action = -1
        trial_reward = 0.0
        trial_correct = False
        current_trial = info.get("trial_index", 0)

        while True:
            phase = info.get("phase")
            phase_step = info.get("phase_step", 0)

            if phase == "response" and phase_step == 0:
                contrast = -float(obs.get("contrast", 0.0))
                block_prior = float(info.get("block_prior", 0.5))
                prev_left = 0.5 if prev_action < 0 else (1.0 if prev_action == ACTION_LEFT else 0.0)
                prev_right = 0.5 if prev_action < 0 else (1.0 if prev_action == ACTION_RIGHT else 0.0)
                feature = torch.tensor(
                    [
                        contrast,
                        abs(contrast),
                        block_prior * self.config.prior_feature_scale,
                        (block_prior - 0.5) * self.config.prior_feature_scale,
                        prev_left * self.config.history_feature_scale,
                        prev_right * self.config.history_feature_scale,
                        (prev_reward - 0.5) * self.config.history_feature_scale,
                        ((1.0 if prev_correct else 0.0) - 0.5) * self.config.history_feature_scale,
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    outputs, hidden = self.model.step(feature, hidden)

                choice_prob = outputs.choice_prob.squeeze()
                if stochastic:
                    dist = torch.distributions.Categorical(logits=torch.stack([1 - choice_prob, choice_prob], dim=-1))
                    sampled_action = dist.sample().item()
                else:
                    sampled_action = ACTION_RIGHT if choice_prob > 0.5 else ACTION_LEFT
                rt_ms = float(outputs.non_decision.squeeze().cpu()) * 1000.0 + self.config.motor_delay_ms
                delay_steps = math.ceil(max(0.0, rt_ms) / step_ms)
                delay_steps = min(delay_steps, response_window - 1)

                pending_action = sampled_action
                action_delay = delay_steps

            action_to_env = ACTION_NO_OP
            if phase == "response" and pending_action != ACTION_NO_OP:
                if action_delay <= 0:
                    action_to_env = pending_action
                    trial_action = pending_action
                    pending_action = ACTION_NO_OP
                    action_delay = 0
                else:
                    action_delay -= 1

            obs, reward, terminated, truncated, info = env.step(action_to_env)

            if reward is not None:
                trial_reward = float(reward)
                trial_correct = reward > 0

            next_trial = info.get("trial_index", current_trial)
            if next_trial != current_trial:
                prev_action = trial_action
                prev_reward = trial_reward
                prev_correct = trial_correct
                trial_action = -1
                trial_reward = 0.0
                trial_correct = False
                current_trial = next_trial

            if terminated or truncated:
                break

        env.close()
        return log_path

    def _rollout_rdm(self, total_trials: int, seed_value: int, run_name: str, log_path: Path, stochastic: bool, balanced: bool) -> Path:
        env_config = RDMConfig(
            trials_per_episode=total_trials,
            include_history=True,
            log_path=log_path,
            agent=RDMAgentMetadata(name="r_ddm", version="prototype"),
            seed=seed_value,
        )
        env = RDMMacaqueEnv(env_config)
        obs, info = env.reset(seed=seed_value)
        current_trial = 0

        if balanced:
            forced_side = (current_trial % 2) # 0=left, 1=right
            obs, info = env.reset(forced_side=forced_side)

        device = next(self.model.parameters()).device
        hidden: torch.Tensor | None = None

        prev_action = -1
        prev_reward = 0.0
        prev_correct = False

        response_phase = next(p for p in env_config.phase_schedule if p.name == "response")
        response_window = response_phase.duration_steps
        step_ms = env_config.step_ms

        pending_action = RDM_ACTION_HOLD
        action_delay = 0
        trial_action = -1
        trial_reward = 0.0
        trial_correct = False
        current_trial = info.get("trial_index", 0)

        while True:
            phase = info.get("phase")
            phase_step = info.get("phase_step", 0)

            if phase == "response" and phase_step == 0:
                coherence = float(obs.get("coherence", 0.0))
                block_prior = 0.5
                prev_left = 0.5 if prev_action < 0 else (1.0 if prev_action == RDM_ACTION_LEFT else 0.0)
                prev_right = 0.5 if prev_action < 0 else (1.0 if prev_action == RDM_ACTION_RIGHT else 0.0)
                feature = torch.tensor(
                    [
                        coherence,
                        abs(coherence),
                        block_prior * self.config.prior_feature_scale,
                        (block_prior - 0.5) * self.config.prior_feature_scale,
                        prev_left * self.config.history_feature_scale,
                        prev_right * self.config.history_feature_scale,
                        (prev_reward - 0.5) * self.config.history_feature_scale,
                        ((1.0 if prev_correct else 0.0) - 0.5) * self.config.history_feature_scale,
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    outputs, hidden = self.model.step(feature, hidden)

                choice_prob = outputs.choice_prob.squeeze()
                if stochastic:
                    dist = torch.distributions.Categorical(logits=torch.stack([1 - choice_prob, choice_prob], dim=-1))
                    sampled_action = dist.sample().item()
                    sampled_action = RDM_ACTION_RIGHT if sampled_action == 1 else RDM_ACTION_LEFT
                else:
                    sampled_action = RDM_ACTION_RIGHT if choice_prob > 0.5 else RDM_ACTION_LEFT
                rt_ms = float(outputs.non_decision.squeeze().cpu()) * 1000.0 + self.config.motor_delay_ms
                delay_steps = math.ceil(max(0.0, rt_ms) / step_ms)
                delay_steps = min(delay_steps, response_window - 1)

                pending_action = sampled_action
                action_delay = delay_steps

            action_to_env = RDM_ACTION_HOLD
            if phase == "response" and pending_action != RDM_ACTION_HOLD:
                if action_delay <= 0:
                    action_to_env = pending_action
                    trial_action = pending_action
                    pending_action = RDM_ACTION_HOLD
                    action_delay = 0
                else:
                    action_delay -= 1

            obs, reward, terminated, truncated, info = env.step(action_to_env)

            if reward is not None:
                trial_reward = float(reward)
                trial_correct = reward > 0

            next_trial = info.get("trial_index", current_trial)
            if next_trial != current_trial:
                prev_action = trial_action
                prev_reward = trial_reward
                prev_correct = trial_correct
                trial_action = -1
                trial_reward = 0.0
                trial_correct = False
                current_trial = next_trial

            if terminated or truncated:
                break

        env.close()
        return log_path
