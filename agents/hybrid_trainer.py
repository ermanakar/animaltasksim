"""Trainer and rollout logic for the hybrid DDM + LSTM agent."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, List
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from agents.losses import (
    LossWeights, choice_loss, history_penalty,
    history_supervision_loss, non_decision_supervision_loss,
    rt_loss, soft_rt_penalty
)
from agents.wfpt_loss import wfpt_loss
from envs.ibl_2afc import (
    ACTION_NO_OP, AgentMetadata as IBLAgentMetadata, IBL2AFCConfig, IBL2AFCEnv
)
from envs.rdm_macaque import (
    ACTION_HOLD, ACTION_LEFT, ACTION_RIGHT, AgentMetadata, RDMConfig, RDMMacaqueEnv
)
from envs.utils_timing import PhaseTiming
from eval.metrics import load_trials
from animaltasksim.seeding import seed_everything

from agents.hybrid_config import HybridDDMPaths, HybridTrainingConfig, SessionBatch
from agents.hybrid_model import HybridDDMModel

class HybridDDMTrainer:
    """Train and deploy the hybrid DDM + LSTM agent."""

    def __init__(self, config: HybridTrainingConfig) -> None:
        self.config = config
        self.config.loss_weights.clamp_non_negative()
        self.device = torch.device("cpu")
        seed_everything(self.config.seed)
        self.sessions = self._load_reference_sessions()
        if not self.sessions:
            raise RuntimeError("No reference sessions found for training.")
        self.feature_dim = self.sessions[0].features.shape[1]
        self.model = HybridDDMModel(
            self.feature_dim,
            self.config.hidden_size,
            self.device,
            drift_scale=self.config.drift_scale,
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    # ------------------------------------------------------------------
    # Reference data handling
    # ------------------------------------------------------------------
    def _load_reference_sessions(self) -> list[SessionBatch]:
        df = load_trials(self.config.reference_log)
        if df.empty:
            return []
        task_key = "ibl_2afc" if self.config.task == "ibl_2afc" else "rdm"
        df = df[df["task"] == task_key].copy()
        if df.empty:
            return []
        df.sort_values(["session_id", "trial_index"], inplace=True)

        sessions: list[SessionBatch] = []
        
        # CRITICAL FIX: Split data into mini-batches for more frequent gradient updates
        # Previously: 1 session → 1 update per epoch → 15 total updates → NO LEARNING
        # Now: Split into chunks → many updates per epoch → proper gradient descent
        batch_size = self.config.max_trials_per_session if self.config.max_trials_per_session else 100
        
        for session_id, group in df.groupby("session_id", sort=False):
            trials = group.copy()
            
            # Split this session into multiple mini-batches
            n_trials = len(trials)
            n_batches = max(1, (n_trials + batch_size - 1) // batch_size)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_trials)
                batch_trials = trials.iloc[start_idx:end_idx]
                
                if batch_trials.empty:
                    continue

                features, choice, choice_mask, rt_ms, rt_mask, correct = self._session_to_arrays(batch_trials)
                win_stay, lose_shift = self._session_history_stats(batch_trials)
                twin_params = self._fit_twin_parameters(choice, choice_mask, rt_ms, rt_mask)
                rt_targets, rt_variances = self._compute_rt_targets(rt_ms, rt_mask, features)
                sessions.append(
                    SessionBatch(
                        features=features,
                        choice=choice,
                        choice_mask=choice_mask,
                        rt_ms=rt_ms,
                        rt_mask=rt_mask,
                        correct=correct,
                        win_stay_target=float(win_stay),
                        lose_shift_target=float(lose_shift),
                        twin_params=twin_params,
                        rt_targets=rt_targets,
                        rt_variances=rt_variances,
                    )
                )
                if self.config.max_sessions is not None and len(sessions) >= self.config.max_sessions:
                    break
            if self.config.max_sessions is not None and len(sessions) >= self.config.max_sessions:
                break
        return sessions

    @staticmethod
    def _map_prev_action(action: str | None) -> float:
        if action == "right":
            return 1.0
        if action == "left":
            return -1.0
        return 0.0

    def _session_to_arrays(self, trials) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feature_rows: List[List[float]] = []
        choice: List[float] = []
        choice_mask: List[float] = []
        rt_ms: List[float] = []
        rt_mask: List[float] = []
        correct: List[float] = []

        max_idx = max(int(trials["trial_index"].max()), 1)

        for _, row in trials.iterrows():
            stim_key = "stimulus_contrast" if self.config.task == "ibl_2afc" else "stimulus_coherence"
            coherence = float(row.get(stim_key, 0.0))
            sign = np.sign(coherence)
            abs_coh = abs(coherence)
            prev_action_val = self._map_prev_action(row.get("prev_action"))
            # Handle prev_reward which may be None or NaN
            prev_reward_raw = row.get("prev_reward", 0.0)
            if prev_reward_raw is None or (isinstance(prev_reward_raw, float) and np.isnan(prev_reward_raw)):
                prev_reward = 0.0
            else:
                prev_reward = float(prev_reward_raw)
            prev_correct = 1.0 if bool(row.get("prev_correct", False)) else 0.0
            trial_index_norm = float(row.get("trial_index", 0)) / max_idx

            feature_rows.append(
                [
                    coherence,
                    abs_coh,
                    sign,
                    prev_action_val,
                    prev_reward,
                    prev_correct,
                    trial_index_norm,
                ]
            )

            action = row.get("action")
            if action == "right":
                choice.append(1.0)
                choice_mask.append(1.0)
            elif action == "left":
                choice.append(0.0)
                choice_mask.append(1.0)
            else:
                choice.append(0.5)
                choice_mask.append(0.0)  # ignore hold trials in supervised loss

            rt_val = row.get("rt_ms")
            if rt_val is None or (isinstance(rt_val, float) and np.isnan(rt_val)):
                rt_ms.append(0.0)
                rt_mask.append(0.0)
            else:
                rt_ms.append(float(rt_val))
                rt_mask.append(1.0)

            correct.append(1.0 if bool(row.get("correct", False)) else 0.0)

        features_np = np.asarray(feature_rows, dtype=np.float32)
        choice_np = np.asarray(choice, dtype=np.float32)
        choice_mask_np = np.asarray(choice_mask, dtype=np.float32)
        rt_np = np.asarray(rt_ms, dtype=np.float32)
        rt_mask_np = np.asarray(rt_mask, dtype=np.float32)
        correct_np = np.asarray(correct, dtype=np.float32)
        return features_np, choice_np, choice_mask_np, rt_np, rt_mask_np, correct_np

    def _fit_twin_parameters(
        self,
        choice: np.ndarray,
        choice_mask: np.ndarray,
        rt_ms: np.ndarray,
        rt_mask: np.ndarray,
    ) -> dict[str, float]:
        valid = (choice_mask > 0.5) & (rt_mask > 0.5) & (rt_ms > 1.0)
        if int(valid.sum()) < 10:
            return {
                "drift": 6.0,
                "bound": 2.5,
                "bias": 0.0,
                "non_decision": 250.0,
                "noise": 1.0,
            }

        y = torch.from_numpy(np.where(choice[valid] > 0.5, 1.0, 0.0)).float().to(self.device)
        rt = torch.from_numpy(rt_ms[valid]).float().to(self.device)

        drift_p = torch.nn.Parameter(torch.tensor(2.0))
        bound_p = torch.nn.Parameter(torch.tensor(0.7))
        bias_p = torch.nn.Parameter(torch.tensor(0.0))
        nd_p = torch.nn.Parameter(torch.tensor(200.0))
        noise_p = torch.nn.Parameter(torch.tensor(0.0))

        optim = torch.optim.Adam([drift_p, bound_p, bias_p, nd_p, noise_p], lr=0.03)

        for _ in range(400):
            optim.zero_grad()
            drift = torch.nn.functional.softplus(drift_p) + 1e-3
            bound = torch.nn.functional.softplus(bound_p) + 0.5
            bias = torch.tanh(bias_p)
            non_decision = torch.nn.functional.softplus(nd_p) + 120.0
            noise = torch.nn.functional.softplus(noise_p) + 1e-3

            loss = wfpt_loss(
                choice=y,
                rt_ms=rt,
                drift=drift.expand_as(y),
                bound=bound.expand_as(y),
                bias=bias.expand_as(y),
                noise=noise.expand_as(y),
                non_decision_ms=non_decision.expand_as(y),
                weight=1.0,
            )
            if not torch.isfinite(loss):
                break
            loss.backward()
            optim.step()

        with torch.no_grad():
            drift_val = float((torch.nn.functional.softplus(drift_p) + 1e-3).cpu().item())
            bound_val = float((torch.nn.functional.softplus(bound_p) + 0.5).cpu().item())
            bias_val = float(torch.tanh(bias_p).cpu().item())
            non_decision_val = float((torch.nn.functional.softplus(nd_p) + 120.0).cpu().item())
            noise_val = float((torch.nn.functional.softplus(noise_p) + 1e-3).cpu().item())

        return {
            "drift": drift_val,
            "bound": bound_val,
            "bias": bias_val,
            "non_decision": non_decision_val,
            "noise": noise_val,
        }

    def _compute_rt_targets(
        self,
        rt_ms: np.ndarray,
        rt_mask: np.ndarray,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = (rt_mask > 0.5) & (rt_ms > 1.0)
        fallback_mean = 1200.0 if self.config.task == "ibl_2afc" else 750.0
        fallback_var = 8000000.0 if self.config.task == "ibl_2afc" else 30000.0
        if not valid.any():
            length = len(rt_ms)
            return (
                np.full(length, fallback_mean, dtype=np.float32),
                np.full(length, fallback_var, dtype=np.float32),
            )

        # coherence stored in first feature entry (signed); use absolute value
        coherences = np.abs(features[:, 0])
        unique = np.unique(coherences[valid])
        unique.sort()

        # Precompute mean/var from reference data (task-specific)
        if self.config.task == "ibl_2afc":
            ref_means = {
                0.0: 2253.21,
                0.0625: 1429.98,
                0.125: 1060.71,
                0.25: 953.88,
                1.0: 652.04,
            }
            ref_vars = {
                0.0: 15831397.23,
                0.0625: 8756352.73,
                0.125: 4891653.08,
                0.25: 6004388.71,
                1.0: 2191865.08,
            }
            default_mean = 1200.0
            default_var = 8000000.0
        else:
            ref_means = {
                0.0: 785.3410672853828,
                0.032: 778.6422018348624,
                0.064: 736.3586206896551,
                0.128: 666.9172413793103,
                0.256: 559.9678899082569,
                0.512: 464.41324200913243,
            }
            ref_vars = {
                0.0: 36552.229381763456,
                0.032: 38897.03253193174,
                0.064: 29724.092081856124,
                0.128: 23129.02763476932,
                0.256: 12263.618235997053,
                0.512: 8138.849779987095,
            }
            default_mean = 750.0
            default_var = 30000.0

        targets = np.empty_like(rt_ms, dtype=np.float32)
        variances = np.empty_like(rt_ms, dtype=np.float32)
        for coh in unique:
            mask = coherences == coh
            targets[mask] = ref_means.get(float(coh), default_mean)
            variances[mask] = ref_vars.get(float(coh), default_var)
        # For coherences not seen in reference, fall back to defaults
        remaining = ~np.isfinite(targets)
        if remaining.any():
            targets[remaining] = default_mean
            variances[remaining] = default_var
        return targets, variances

    @staticmethod
    def _session_history_stats(trials) -> tuple[float, float]:
        win_stay_events = 0
        win_stay_total = 0
        lose_shift_events = 0
        lose_shift_total = 0
        prev_action = None
        prev_correct = None
        for _, row in trials.iterrows():
            action = row.get("action")
            correct = bool(row.get("correct", False))
            if prev_action in {"left", "right"}:
                if prev_correct:
                    win_stay_total += 1
                    if action == prev_action:
                        win_stay_events += 1
                else:
                    lose_shift_total += 1
                    if action and prev_action and action != prev_action:
                        lose_shift_events += 1
            prev_action = action if action in {"left", "right"} else prev_action
            prev_correct = correct
        win_stay_rate = win_stay_events / max(win_stay_total, 1)
        lose_shift_rate = lose_shift_events / max(lose_shift_total, 1)
        return win_stay_rate, lose_shift_rate

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, loss_weights: LossWeights | None = None) -> dict[str, list[float]]:
        """Train for config.epochs using given or default loss weights."""
        weights = loss_weights if loss_weights is not None else self.config.loss_weights
        metrics: dict[str, list[float]] = {
            "epoch_choice_loss": [],
            "epoch_rt_loss": [],
            "epoch_soft_rt_penalty": [],
            "epoch_history_penalty": [],
            "epoch_history_supervision": [],
            "epoch_drift_supervision": [],
            "epoch_non_decision_supervision": [],
            "epoch_drift_magnitude": [],
            "epoch_wfpt_loss": [],
            "epoch_twin_supervision": [],
            "noise": [],
            "mean_bound": [],
        }
        for epoch in range(self.config.epochs):
            epoch_choice = 0.0
            epoch_rt = 0.0
            epoch_soft_rt = 0.0
            epoch_hist = 0.0
            epoch_hist_sup = 0.0
            epoch_drift_sup = 0.0
            epoch_non_decision_sup = 0.0
            epoch_drift_mag = 0.0
            epoch_wfpt = 0.0
            epoch_twin_sup = 0.0
            session_count = 0
            # Shuffle sessions using random.shuffle for type compatibility
            shuffled_sessions = list(self.sessions)
            random.shuffle(shuffled_sessions)
            for session in shuffled_sessions:
                self.optimizer.zero_grad()
                h, c = self.model.init_state()
                features = torch.from_numpy(session.features).to(self.device)
                choice = torch.from_numpy(session.choice).to(self.device)
                choice_mask = torch.from_numpy(session.choice_mask).to(self.device)
                rt_ms = torch.from_numpy(session.rt_ms).to(self.device)
                rt_mask = torch.from_numpy(session.rt_mask).to(self.device)

                total_choice_loss = torch.zeros(1, device=self.device)
                total_rt_loss = torch.zeros(1, device=self.device)
                choice_weight = 0.0
                rt_weight = 0.0

                prob_buffer: list[float] = []
                prob_tensor_buffer: list[torch.Tensor] = []  # differentiable copy
                history_prob_tensor_buffer: list[torch.Tensor] = []  # gradient-isolated for history loss
                stay_tendency_buffer: list[torch.Tensor] = []  # from history_network
                drift_gain_buffer: list[torch.Tensor] = []  # Collect for supervision
                bound_buffer: list[torch.Tensor] = []
                bias_buffer: list[torch.Tensor] = []
                noise_buffer: list[torch.Tensor] = []
                soft_rt_pred_buffer: list[torch.Tensor] = []
                soft_rt_target_buffer: list[torch.Tensor] = []
                soft_rt_var_buffer: list[torch.Tensor] = []
                soft_rt_mask_buffer: list[torch.Tensor] = []
                soft_rt_weights_buffer: list[torch.Tensor] = []

                # Buffers for WFPT loss (collect all DDM params + observed choice/RT)
                non_decision_buffer: list[torch.Tensor] = []
                wfpt_choice_buffer: list[torch.Tensor] = []
                wfpt_rt_buffer: list[torch.Tensor] = []
                wfpt_drift_buffer: list[torch.Tensor] = []
                wfpt_bound_buffer: list[torch.Tensor] = []
                wfpt_bias_buffer: list[torch.Tensor] = []
                wfpt_noise_buffer: list[torch.Tensor] = []
                wfpt_nondecision_buffer: list[torch.Tensor] = []
                
                # Truncated BPTT: detach every N trials to prevent gradient explosion
                tbptt_chunk_size = 20

                for idx in range(features.shape[0]):
                    # Detach states periodically for truncated BPTT
                    if idx > 0 and idx % tbptt_chunk_size == 0:
                        h = h.detach()
                        c = c.detach()
                    x = features[idx : idx + 1]
                    out, (h, c) = self.model(x, (h, c))
                    non_decision_buffer.append(out["non_decision_ms"])
                    coherence = x[0, 0]
                    drift = out["drift_gain"] * coherence
                    bound = out["bound"]
                    noise = out["noise"]

                    # Collect drift_gain for supervision loss
                    drift_gain_buffer.append(out["drift_gain"])
                    bound_buffer.append(bound)
                    bias_buffer.append(out["bias"])
                    noise_buffer.append(noise)

                    # Exact DDM choice probability with starting-point bias.
                    # P(right) = [1 - exp(-2vz/σ²)] / [1 - exp(-2va/σ²)]
                    # where v=drift, z=bound+bias (distance from lower boundary),
                    # a=2*bound (total separation), σ=noise.
                    # At v→0:  P(right) = z/a = (bound + bias) / (2*bound).
                    # Combine with history effects immediately for true joint learning
                    bias_val = out["bias"]  # tanh output ∈ (-1, 1)
                    history_bias_val = out["history_bias"]  # tanh output ∈ (-1, 1)
                    
                    stay_tendency_val = out["stay_tendency"]
                    prev_action_val_t = x[0, 3] # features array index 3 is prev_action_val
                    prev_direction = torch.where(
                        prev_action_val_t > 0, 
                        torch.tensor(1.0, device=self.device), 
                        torch.where(prev_action_val_t < 0, torch.tensor(-1.0, device=self.device), torch.tensor(0.0, device=self.device))
                    )
                    
                    history_drift = stay_tendency_val * self.model.history_drift_scale * prev_direction
                    
                    # Attention gate: suppress history drift when stimulus contrast is high
                    confidence = torch.clamp(torch.abs(coherence), min=0.0, max=1.0)
                    gated_history_drift = history_drift * (1.0 - confidence)
                    
                    effective_drift = drift + gated_history_drift
                    
                    stay_shift = stay_tendency_val * self.model.history_bias_scale * bound
                    effective_bias = bias_val + stay_shift * prev_direction
                    
                    # ------------------------------------------------------------------
                    # Differentiable DDM Simulation (Replaces Analytical Solver)
                    # ------------------------------------------------------------------
                    # To solve the Decoupling Problem mathematically, the agent must feel
                    # the literal gradient of taking too many steps in the DDM. Analytical
                    # solutions (tanh(kappa)) allow the agent to cheat the gradients by 
                    # pushing bound->infinity and drift->0. We prevent this by unrolling 
                    # the Euler-Maruyama simulation directly in PyTorch, creating a 
                    # computational graph from the reaction time back to the LSTM outputs.

                    dt = self.config.step_ms / 1000.0
                    sqrt_dt = np.sqrt(dt)
                    max_steps = self.config.max_commit_steps
                    
                    # 1. Generate stochastic noise for the trial
                    # (Batch size 1, sequence length max_steps)
                    step_noise = torch.randn(max_steps, device=self.device) * noise * sqrt_dt
                    
                    # 2. Add constant drift to each step
                    step_drift = effective_drift * dt
                    
                    # 3. Accumulate evidence starting from bias
                    evidence_trajectory = effective_bias + torch.cumsum(step_drift + step_noise, dim=0)
                    
                    # 4. Find the first step where evidence crosses the upper or lower bound
                    # Softmax temperature controls how harshly we enforce the boundary crossing
                    temp = 0.1
                    
                    # Probability of crossing upper bound at each step
                    prob_upper = torch.sigmoid((evidence_trajectory - bound) / temp)
                    # Probability of crossing lower bound at each step
                    prob_lower = torch.sigmoid((-bound - evidence_trajectory) / temp)
                    
                    # Probability of committing at each step
                    prob_commit = torch.clamp(prob_upper + prob_lower, 0.0, 1.0)
                    
                    # Find exactly when we cross (first step where prob_commit -> 1)
                    # We compute the cumulative product of NOT committing, which drops to 0 
                    # after the first cross. 
                    prob_not_commit = 1.0 - prob_commit
                    cum_not_commit = torch.cat([
                        torch.ones(1, device=self.device), 
                        torch.cumprod(prob_not_commit[:-1], dim=0)
                    ])
                    
                    # Density function of commit time 
                    commit_density = prob_commit * cum_not_commit
                    
                    # Expected number of steps to commit
                    step_indices = torch.arange(1, max_steps + 1, device=self.device, dtype=torch.float)
                    expected_steps = torch.sum(step_indices * commit_density)
                    
                    # If it never crosses the bound, add a massive penalty to expected_steps
                    prob_timeout = cum_not_commit[-1] * (1.0 - prob_commit[-1])
                    expected_steps = expected_steps + prob_timeout * max_steps * 10.0
                    
                    predicted_rt = out["non_decision_ms"] + expected_steps * self.config.step_ms
                    predicted_rt = torch.clamp(
                        predicted_rt,
                        min=float(self.config.step_ms * self.config.min_commit_steps)
                    )

                    # For choice probability, we compute the ratio of upper-bound crossings 
                    # to all boundary crossings.
                    p_right_given_commit = torch.sum(prob_upper * cum_not_commit) / torch.clamp(torch.sum(commit_density), min=1e-8)
                    
                    # If it times out, the choice is determined by the final evidence state
                    p_right_given_timeout = torch.sigmoid(evidence_trajectory[-1] / temp)
                    
                    prob_right = (1.0 - prob_timeout) * p_right_given_commit + prob_timeout * p_right_given_timeout

                    # Lapse is applied only in rollout, NOT in training. The reference
                    # animal data already contains the animal's own lapse; blending
                    # lapse into training prob_right double-counts it, compresses choice
                    # gradients, and flattens the psychometric curve (sweep_fixed_lapse_v1
                    # showed psych slope stuck at ~8.5 regardless of drift_scale).

                    prob_right = torch.clamp(prob_right, 1e-6, 1.0 - 1e-6).unsqueeze(0)

                    # Collect per-trial values for downstream history losses.
                    # These were accidentally dropped during the analytical→simulation
                    # refactor. Without them, history/per-trial-history losses are
                    # silently zero (empty buffers).
                    prob_buffer.append(float(prob_right.detach().cpu()))
                    prob_tensor_buffer.append(prob_right)
                    stay_tendency_buffer.append(out["stay_tendency"])
                    history_logit = history_bias_val * self.model.history_bias_scale * 4.0
                    history_prob_stay = torch.sigmoid(history_logit)
                    history_prob_tensor_buffer.append(history_prob_stay)

                    if (
                        weights.rt_soft > 0.0
                        and session.rt_targets is not None
                        and session.rt_variances is not None
                    ):
                        soft_rt_pred_buffer.append(predicted_rt)
                        soft_rt_target_buffer.append(
                            torch.tensor(
                                [session.rt_targets[idx]], device=self.device, dtype=torch.float32
                            )
                        )
                        soft_rt_var_buffer.append(
                            torch.tensor(
                                [session.rt_variances[idx]], device=self.device, dtype=torch.float32
                            )
                        )
                        soft_rt_mask_buffer.append(rt_mask[idx : idx + 1])

                        # Add per-coherence weighting
                        coherence_val = torch.abs(features[idx, 0])
                        # Inverse relationship: higher weight for lower coherence
                        weight = 1.0 / (1.0 + 2.0 * coherence_val)
                        soft_rt_weights_buffer.append(weight.unsqueeze(0))

                    if choice_mask[idx] > 0:
                        loss_c = choice_loss(prob_right, choice[idx : idx + 1], choice_mask[idx : idx + 1])
                        total_choice_loss = total_choice_loss + loss_c
                        choice_weight += float(choice_mask[idx].detach().cpu())
                    if rt_mask[idx] > 0:
                        loss_r = rt_loss(predicted_rt, rt_ms[idx : idx + 1], rt_mask[idx : idx + 1])
                        total_rt_loss = total_rt_loss + loss_r
                        rt_weight += float(rt_mask[idx].detach().cpu())
                    
                    # Collect for WFPT loss (only for valid choice+RT trials)
                    if choice_mask[idx] > 0 and rt_mask[idx] > 0:
                        # Convert choice from {-1, 1} or {0, 1} to {0, 1}
                        choice_binary = (choice[idx] + 1.0) / 2.0 if choice[idx] < 0.5 else choice[idx]
                        wfpt_choice_buffer.append(choice_binary.unsqueeze(0))
                        wfpt_rt_buffer.append(rt_ms[idx].unsqueeze(0))
                        wfpt_drift_buffer.append(effective_drift.unsqueeze(0))
                        wfpt_bound_buffer.append(bound.unsqueeze(0))
                        wfpt_bias_buffer.append(effective_bias.unsqueeze(0))
                        wfpt_noise_buffer.append(noise.unsqueeze(0))
                        wfpt_nondecision_buffer.append(out["non_decision_ms"].unsqueeze(0))

                if choice_weight > 0:
                    total_choice_loss = total_choice_loss / choice_weight
                if rt_weight > 0:
                    total_rt_loss = total_rt_loss / rt_weight

                total_loss = (
                    weights.choice * total_choice_loss
                    + weights.rt * total_rt_loss
                )

                if weights.history > 0.0:
                    pred_win_stay, pred_lose_shift = self._estimate_history(prob_buffer, session)
                    history_vec = torch.tensor([pred_win_stay, pred_lose_shift], device=self.device)
                    target_vec = torch.tensor(
                        [session.win_stay_target, session.lose_shift_target], device=self.device
                    )
                    mask = torch.tensor([1.0, 1.0], device=self.device)
                    hist_loss = history_penalty(history_vec, target_vec, mask)
                    total_loss = total_loss + weights.history * hist_loss
                else:
                    hist_loss = torch.zeros(1, device=self.device)

                hist_sup_loss = torch.zeros(1, device=self.device)
                if weights.history_supervision > 0.0:
                    pred_win_stay, pred_lose_shift = self._estimate_history(prob_buffer, session)
                    hist_sup_loss = history_supervision_loss(
                        pred_win_stay=torch.tensor(pred_win_stay, device=self.device),
                        pred_lose_shift=torch.tensor(pred_lose_shift, device=self.device),
                        target_win_stay=torch.tensor(session.win_stay_target, device=self.device),
                        target_lose_shift=torch.tensor(session.lose_shift_target, device=self.device),
                    )
                    total_loss = total_loss + weights.history_supervision * hist_sup_loss

                # Per-trial history loss using the SEPARATE HISTORY NETWORK.
                # stay_tendency comes from history_network(prev_action, prev_reward),
                # which bypasses the LSTM entirely. Convert to P(stay) via sigmoid
                # for the training loss; in rollout, stay_tendency shifts the DDM
                # starting point directly.
                pt_hist_loss = torch.zeros(1, device=self.device)
                if weights.per_trial_history > 0.0 and stay_tendency_buffer:
                    stay_tendency_tensor = torch.cat(stay_tendency_buffer).squeeze()
                    # Convert stay_tendency to P(stay) via sigmoid using learned scale
                    stay_prob = torch.sigmoid(stay_tendency_tensor * self.model.history_bias_scale * 4.0)
                    # Extract prev_action and prev_reward from features
                    prev_act = torch.from_numpy(session.features[:, 3]).to(self.device)
                    prev_rew = torch.from_numpy(session.features[:, 4]).to(self.device)
                    choice_m = torch.from_numpy(session.choice_mask).bool().to(self.device)
                    n = min(len(stay_prob), len(prev_act), len(choice_m))
                    stay_prob = stay_prob[:n]
                    prev_act = prev_act[:n]
                    prev_rew = prev_rew[:n]
                    choice_m = choice_m[:n]
                    # Valid = has a previous action and is a commit trial
                    valid = prev_act.ne(0.0) & choice_m
                    win_mask = valid & (prev_rew > 0.5)
                    lose_mask = valid & (prev_rew <= 0.5)
                    target_ws = float(session.win_stay_target)
                    target_ls = float(session.lose_shift_target)
                    if win_mask.any():
                        # Win: push P(stay) toward target_win_stay
                        target = torch.full_like(stay_prob[win_mask], target_ws)
                        pt_hist_loss = pt_hist_loss + F.mse_loss(stay_prob[win_mask], target)
                    if lose_mask.any():
                        # Lose: push P(shift)=1-P(stay) toward target_lose_shift
                        shift_prob = 1.0 - stay_prob
                        target = torch.full_like(shift_prob[lose_mask], target_ls)
                        pt_hist_loss = pt_hist_loss + F.mse_loss(shift_prob[lose_mask], target)
                    total_loss = total_loss + weights.per_trial_history * pt_hist_loss

                # Drift supervision: penalize weak drift parameters
                drift_sup_loss = torch.zeros(1, device=self.device)
                if weights.drift_supervision > 0.0 and drift_gain_buffer:
                    from agents.losses import drift_supervision_loss
                    drift_gains = torch.cat(drift_gain_buffer)
                    drift_sup_loss = drift_supervision_loss(drift_gains, target_gain=5.0)
                    total_loss = total_loss + weights.drift_supervision * drift_sup_loss

                non_decision_sup_loss = torch.zeros(1, device=self.device)
                if weights.non_decision_supervision > 0.0 and non_decision_buffer:
                    non_decisions = torch.cat(non_decision_buffer)
                    non_decision_sup_loss = non_decision_supervision_loss(
                        non_decisions, target_ms=200.0
                    )
                    total_loss = (
                        total_loss + weights.non_decision_supervision * non_decision_sup_loss
                    )

                # Drift magnitude regularization: anchor drift_gain scale to prevent collapse
                drift_mag_loss = torch.zeros(1, device=self.device)
                if weights.drift_magnitude > 0.0 and drift_gain_buffer:
                    drift_gains = torch.cat(drift_gain_buffer)
                    target = self.config.drift_magnitude_target
                    drift_mag_loss = torch.mean((drift_gains - target) ** 2)
                    total_loss = total_loss + weights.drift_magnitude * drift_mag_loss

                # WFPT likelihood loss: statistically correct DDM objective
                wfpt_loss_val = torch.zeros(1, device=self.device)
                if weights.wfpt > 0.0 and wfpt_choice_buffer:
                    wfpt_choices = torch.cat(wfpt_choice_buffer)
                    wfpt_rts = torch.cat(wfpt_rt_buffer)
                    wfpt_drifts = torch.cat(wfpt_drift_buffer)
                    wfpt_bounds = torch.cat(wfpt_bound_buffer)
                    wfpt_biases = torch.cat(wfpt_bias_buffer)
                    wfpt_noises = torch.cat(wfpt_noise_buffer)
                    wfpt_nondecisions = torch.cat(wfpt_nondecision_buffer)

                    wfpt_loss_val = wfpt_loss(
                        choice=wfpt_choices,
                        rt_ms=wfpt_rts,
                        drift=wfpt_drifts,
                        bound=wfpt_bounds,
                        bias=wfpt_biases,
                        noise=wfpt_noises,
                        non_decision_ms=wfpt_nondecisions,
                        weight=1.0,
                    )
                    total_loss = total_loss + weights.wfpt * wfpt_loss_val

                soft_rt_loss = torch.zeros(1, device=self.device)
                if weights.rt_soft > 0.0 and soft_rt_pred_buffer:
                    preds = torch.cat(soft_rt_pred_buffer)
                    targets = torch.cat(soft_rt_target_buffer)
                    variances = torch.cat(soft_rt_var_buffer)
                    masks = torch.cat(soft_rt_mask_buffer)
                    rt_weights = torch.cat(soft_rt_weights_buffer)
                    soft_rt_loss = soft_rt_penalty(preds, targets, variances, masks, rt_weights)
                    total_loss = total_loss + weights.rt_soft * soft_rt_loss

                twin_sup_loss = torch.zeros(1, device=self.device)
                if weights.twin_supervision > 0.0 and session.twin_params and drift_gain_buffer:
                    components: list[torch.Tensor] = []
                    twin = session.twin_params
                    predicted_drift_mean = torch.cat(drift_gain_buffer).mean()
                    components.append(
                        ((predicted_drift_mean - torch.tensor(twin["drift"], device=self.device)) / (abs(twin["drift"]) + 1.0)) ** 2
                    )
                    if bound_buffer:
                        components.append(
                            ((torch.cat(bound_buffer).mean() - torch.tensor(twin["bound"], device=self.device)) / (abs(twin["bound"]) + 1.0)) ** 2
                        )
                    if bias_buffer:
                        components.append(
                            ((torch.cat(bias_buffer).mean() - torch.tensor(twin["bias"], device=self.device)) / 1.0) ** 2
                        )
                    if non_decision_buffer:
                        components.append(
                            ((torch.cat(non_decision_buffer).mean() - torch.tensor(twin["non_decision"], device=self.device)) / (abs(twin["non_decision"]) + 1.0)) ** 2
                        )
                    if noise_buffer:
                        components.append(
                            ((torch.cat(noise_buffer).mean() - torch.tensor(twin["noise"], device=self.device)) / (abs(twin["noise"]) + 1.0)) ** 2
                        )
                    if components:
                        twin_sup_loss = torch.mean(torch.stack(components))
                        total_loss = total_loss + weights.twin_supervision * twin_sup_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                with torch.no_grad():
                    self.model.log_noise.data = torch.nan_to_num(
                        self.model.log_noise.data,
                        nan=0.0,
                        posinf=5.0,
                        neginf=-5.0,
                    )
                    self.model.log_noise.data.clamp_(-5.0, 5.0)

                epoch_choice += float(total_choice_loss.detach().cpu())
                epoch_rt += float(total_rt_loss.detach().cpu())
                epoch_soft_rt += float(soft_rt_loss.detach().cpu())
                epoch_hist += float(hist_loss.detach().cpu())
                epoch_hist_sup += float(hist_sup_loss.detach().cpu())
                epoch_drift_sup += float(drift_sup_loss.detach().cpu())
                epoch_non_decision_sup += float(non_decision_sup_loss.detach().cpu())
                epoch_drift_mag += float(drift_mag_loss.detach().cpu())
                epoch_wfpt += float(wfpt_loss_val.detach().cpu())
                epoch_twin_sup += float(twin_sup_loss.detach().cpu())
                session_count += 1

            metrics["epoch_choice_loss"].append(epoch_choice / max(session_count, 1))
            metrics["epoch_rt_loss"].append(epoch_rt / max(session_count, 1))
            metrics["epoch_soft_rt_penalty"].append(
                epoch_soft_rt / max(session_count, 1)
            )
            metrics["epoch_history_penalty"].append(epoch_hist / max(session_count, 1))
            metrics["epoch_history_supervision"].append(
                epoch_hist_sup / max(session_count, 1)
            )
            metrics["epoch_drift_supervision"].append(epoch_drift_sup / max(session_count, 1))
            metrics["epoch_non_decision_supervision"].append(
                epoch_non_decision_sup / max(session_count, 1)
            )
            metrics["epoch_drift_magnitude"].append(epoch_drift_mag / max(session_count, 1))
            metrics["epoch_wfpt_loss"].append(epoch_wfpt / max(session_count, 1))
            metrics["epoch_twin_supervision"].append(epoch_twin_sup / max(session_count, 1))
            noise_value = torch.exp(torch.nan_to_num(self.model.log_noise, nan=0.0)).detach().cpu()
            metrics["noise"].append(float(noise_value))
            metrics["mean_bound"].append(self._estimate_mean_bound())
        return metrics

    def _estimate_mean_bound(self) -> float:
        with torch.no_grad():
            bounds: list[float] = []
            for session in self.sessions[:5]:
                h, c = self.model.init_state()
                features = torch.from_numpy(session.features).to(self.device)
                for idx in range(min(10, features.shape[0])):
                    out, (h, c) = self.model(features[idx : idx + 1], (h, c))
                    bounds.append(float(out["bound"].cpu().item()))
            return float(np.mean(bounds)) if bounds else 0.0

    def _estimate_history(self, probs: list[float], session: SessionBatch) -> tuple[float, float]:
        """Estimate win-stay/lose-shift from predicted probabilities."""

        if not probs:
            return float(session.win_stay_target), float(session.lose_shift_target)

        probabilities = np.asarray(probs, dtype=np.float32)
        mask = session.choice_mask
        correct = session.correct
        predicted = np.full(probabilities.shape, np.nan, dtype=np.float32)

        for idx, (prob, m) in enumerate(zip(probabilities, mask)):
            if m > 0:
                predicted[idx] = 1.0 if prob >= 0.5 else 0.0

        win_total = 0
        win_events = 0
        lose_total = 0
        lose_events = 0

        for idx in range(1, len(probabilities)):
            if mask[idx] <= 0 or mask[idx - 1] <= 0:
                continue
            prev_choice = predicted[idx - 1]
            curr_choice = predicted[idx]
            if np.isnan(prev_choice) or np.isnan(curr_choice):
                continue
            prev_correct = bool(correct[idx - 1] > 0.5)
            if prev_correct:
                win_total += 1
                if curr_choice == prev_choice:
                    win_events += 1
            else:
                lose_total += 1
                if curr_choice != prev_choice:
                    lose_events += 1

        win_rate = win_events / win_total if win_total > 0 else float(session.win_stay_target)
        lose_rate = lose_events / lose_total if lose_total > 0 else float(session.lose_shift_target)
        return float(win_rate), float(lose_rate)

    # ------------------------------------------------------------------
    # DDM Simulation
    # ------------------------------------------------------------------
    def _simulate_ddm(
        self,
        drift: float,
        bound: float,
        noise: float,
        bias: float = 0.0,
        dt: float = 0.01,
        max_steps: int = 120,
    ) -> tuple[int, int]:
        """Run stochastic DDM simulation via Euler-Maruyama.
        
        Returns:
            (action, num_steps) where action is 0=left, 1=right
        """
        evidence = bias
        sqrt_dt = np.sqrt(dt)
        
        for step in range(max_steps):
            # Euler-Maruyama: dE = drift*dt + noise*sqrt(dt)*N(0,1)
            evidence += drift * dt + noise * sqrt_dt * np.random.randn()
            
            # Check bounds
            if evidence >= bound:
                return ACTION_RIGHT, step + 1
            elif evidence <= -bound:
                return ACTION_LEFT, step + 1
        
        # Timeout: choose based on current evidence
        return (ACTION_RIGHT if evidence > 0 else ACTION_LEFT), max_steps

    # ------------------------------------------------------------------
    # Rollout in environment
    # ------------------------------------------------------------------
    def rollout(self, paths: HybridDDMPaths) -> dict[str, float]:
        # Task-conditional environment setup
        if self.config.task == "ibl_2afc":
            custom_phases = (
                PhaseTiming("iti", 10),
                PhaseTiming("stimulus", 10),
                PhaseTiming("response", self.config.max_commit_steps),
                PhaseTiming("outcome", 10),
            )
            ibl_config = IBL2AFCConfig(
                trials_per_episode=self.config.trials_per_episode,
                log_path=paths.log,
                agent=IBLAgentMetadata(name="hybrid_ddm", version=self.config.agent_version),
                seed=self.config.seed,
                phase_schedule=custom_phases,
            )
            env = IBL2AFCEnv(ibl_config)
            wait_action = ACTION_NO_OP
            new_trial_phase = "iti"
        else:
            rdm_config = RDMConfig(
                trials_per_episode=self.config.trials_per_episode,
                log_path=paths.log,
                agent=AgentMetadata(name="hybrid_ddm", version=self.config.agent_version),
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
        # Cap max_commit_steps to env's response phase duration to prevent
        # the agent from planning commits beyond the response window.
        response_phase = next(p for p in env._phase_schedule if p.name == "response")  # noqa: SLF001
        effective_max_commit = min(self.config.max_commit_steps, response_phase.duration_steps)
        metrics: dict[str, list[float]] = {
            "cumulative_reward": [],
            "mean_rt_ms": [],
        }
        for episode in range(self.config.episodes):
            observation, info = env.reset(seed=self.config.seed + episode)
            h, c = self.model.init_state()
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
                        trial_norm = float(trial_idx) / max(
                            self.config.trials_per_episode, 1
                        )
                        features = self._features_from_trial(
                            current_stimulus,
                            prev_action_val,
                            prev_reward,
                            prev_correct,
                            trial_norm,
                        )
                        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
                        out, (h, c) = self.model(x, (h, c))

                        # Extract DDM parameters
                        stimulus = x[0, 0].item()
                        drift_gain = out["drift_gain"].item()
                        bound = out["bound"].item()
                        noise = out["noise"].item()
                        bias = out["bias"].item()
                        stay_tendency = out["stay_tendency"].item()
                        non_decision_ms = out["non_decision_ms"].item()

                        # Combine DDM bias with history-dependent stay bias
                        prev_direction = 1.0 if prev_action_val == 1 else (-1.0 if prev_action_val == -1 else 0.0)
                        # Starting-point bias: affects ambiguous trials
                        stay_shift = stay_tendency * self.config.history_bias_scale * bound
                        combined_start = bias + stay_shift * prev_direction
                        # Drift-rate bias: affects ALL trials including high-coherence
                        history_drift = stay_tendency * self.config.history_drift_scale * prev_direction
                        
                        # Attention gate: suppress history drift when stimulus contrast is high
                        confidence = min(abs(stimulus), 1.0)
                        gated_history_drift = history_drift * (1.0 - confidence)

                        drift = drift_gain * stimulus + gated_history_drift

                        dt = self.config.step_ms / 1000.0

                        # Stochastic lapse: on a fraction of trials the agent
                        # disengages and guesses randomly, producing realistic
                        # lapse rates (~5% in IBL mice). Fixed rate, not learnable.
                        if random.random() < self.config.lapse_rate:
                            planned_action = random.choice([ACTION_LEFT, ACTION_RIGHT])
                            ddm_steps = random.randint(
                                self.config.min_commit_steps, effective_max_commit
                            )
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
                        commit_step_target = np.clip(
                            commit_step_target,
                            self.config.min_commit_steps,
                            effective_max_commit,
                        )
                    phase_step_val = info.get("phase_step", 0)
                    current_step = int(phase_step_val) if isinstance(phase_step_val, (int, float)) else 0
                    if current_step + 1 >= commit_step_target:
                        action = planned_action
                    else:
                        action = wait_action
                else:
                    action = wait_action
                observation, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += float(reward)
                if info["phase"] == "outcome" and info.get("phase_step", 0) == 0:
                    actual_rt_ms = float((env._rt_steps or 0) * step_ms)  # noqa: SLF001
                    if actual_rt_ms > 0:
                        rt_tracker.append(actual_rt_ms)
                    if planned_action == ACTION_RIGHT:
                        prev_action_val = 1.0
                    elif planned_action == ACTION_LEFT:
                        prev_action_val = -1.0
                    else:
                        prev_action_val = 0.0
                    # Determine correct answer (task-dependent for zero-stimulus)
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

    def _get_stimulus(self, env: Any) -> float:
        """Extract the signed stimulus value from the environment (task-dependent)."""
        if self.config.task == "ibl_2afc":
            return float(env._stimulus.get("contrast", 0.0))  # noqa: SLF001
        return float(getattr(env, "_signed_coherence", 0.0))

    def _features_from_trial(
        self,
        coherence: float,
        prev_action_val: float,
        prev_reward: float,
        prev_correct: float,
        trial_norm: float,
    ) -> np.ndarray:
        sign = np.sign(coherence)
        abs_coh = abs(coherence)
        return np.array(
            [
                coherence,
                abs_coh,
                sign,
                prev_action_val,
                prev_reward,
                prev_correct,
                trial_norm,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Persistence utilities
    # ------------------------------------------------------------------
    def save(self, paths: HybridDDMPaths, training_metrics: dict[str, list[float]], rollout_stats: dict[str, float]) -> None:
        config_payload = asdict(self.config)
        for key, value in list(config_payload.items()):
            if isinstance(value, Path):
                config_payload[key] = str(value)
        config_payload["output_dir"] = str(self.config.output_dir)
        paths.config.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        paths.metrics.write_text(json.dumps({"training": training_metrics, "rollout": rollout_stats}, indent=2), encoding="utf-8")
        torch.save(self.model.state_dict(), paths.model)


def _evaluate_phase_success(
    trials_path: Path,
    criteria: dict[str, float],
    reference_path: Path,
    task: str = "rdm",
) -> tuple[bool, dict[str, float]]:
    """Evaluate whether phase success criteria are met."""
    from scipy import stats as scipy_stats

    from eval.metrics import compute_history_metrics, load_trials

    # Load agent trials
    df = load_trials(trials_path)
    if df.empty:
        return False, {}

    # Filter for valid trials for chronometric analysis (task-dependent stimulus key)
    stim_col = "stimulus_contrast" if task == "ibl_2afc" else "stimulus_coherence"
    filtered = df[(df["rt_ms"].notnull()) & (df[stim_col].notnull())].copy()
    if filtered.empty:
        return False, {}

    rts = filtered["rt_ms"].values.astype(float)
    coherences = filtered[stim_col].values.astype(float)
    abs_coh = np.abs(coherences)

    # Compute RT-coherence metrics
    linreg_result = scipy_stats.linregress(abs_coh, rts)
    slope_val = float(linreg_result.slope)  # type: ignore[attr-defined]
    r_val = float(linreg_result.rvalue)  # type: ignore[attr-defined]
    r2_val = r_val**2
    rt_diff = float(np.abs(rts[abs_coh < 0.05].mean() - rts[abs_coh > 0.5].mean()))

    # Compute history metrics
    history = compute_history_metrics(df)

    metrics = {
        "slope": slope_val,
        "slope_abs": abs(slope_val),
        "r2": r2_val,
        "rt_diff_abs": rt_diff,
        "sticky_choice": history.sticky_choice,
    }

    # Check criteria
    success = True
    if "min_slope_abs" in criteria and metrics["slope_abs"] < criteria["min_slope_abs"]:
        success = False
    if "min_r2" in criteria and metrics["r2"] < criteria["min_r2"]:
        success = False
    if "min_rt_diff_abs" in criteria and metrics["rt_diff_abs"] < criteria["min_rt_diff_abs"]:
        success = False
    if "max_slope" in criteria and metrics["slope"] > criteria["max_slope"]:
        success = False
    if "min_slope" in criteria and metrics["slope"] < criteria["min_slope"]:
        success = False
    if "max_sticky_choice" in criteria and metrics.get("sticky_choice", 1.0) > criteria["max_sticky_choice"]:
        success = False

    return success, metrics


def train_hybrid_curriculum(config: HybridTrainingConfig) -> dict[str, Any]:
    """Train using curriculum learning with phased loss weights."""
    if config.curriculum is None:
        raise ValueError("Curriculum config required for curriculum training")
    
    curriculum = config.curriculum
    trainer = HybridDDMTrainer(config)
    # Ensure commit window is wide enough before applying curriculum adjustments.
    trainer.config.max_commit_steps = max(trainer.config.max_commit_steps, 120)
    paths = config.output_paths()
    
    # Track all phase metrics with proper types
    cumulative_metrics: dict[str, list[float]] = {
        "epoch_choice_loss": [],
        "epoch_rt_loss": [],
        "epoch_history_penalty": [],
        "epoch_drift_supervision": [],
        "noise": [],
        "mean_bound": [],
    }
    phase_results: list[dict] = []
    
    for phase_idx, phase in enumerate(curriculum.phases):
        print(f"\n{'='*80}")
        print(f"Starting {phase.name} (Phase {phase_idx + 1}/{len(curriculum.phases)})")
        print(f"Loss weights: {asdict(phase.loss_weights)}")
        print(f"Epochs: {phase.epochs}")
        print(f"Success criteria: {phase.success_criteria}")

        # Apply per-phase overrides
        original_epochs = trainer.config.epochs
        trainer.config.epochs = phase.epochs
        if phase.min_commit_steps is not None:
            trainer.config.min_commit_steps = phase.min_commit_steps
        if phase.max_commit_steps is not None:
            trainer.config.max_commit_steps = phase.max_commit_steps
        phase_min_commit = trainer.config.min_commit_steps
        phase_max_commit = trainer.config.max_commit_steps
        print(
            f"Commit window (steps): min={phase_min_commit}, max={phase_max_commit}"
        )
        print(f"{'='*80}\n")

        # Optionally freeze all params except history modules for this phase
        original_optimizer = None
        frozen_params: list[tuple[str, torch.nn.Parameter]] = []
        if phase.freeze_except_history_bias:
            # Save original optimizer and freeze non-history params
            original_optimizer = trainer.optimizer
            for name, param in trainer.model.named_parameters():
                # Keep win/lose history networks and history_bias_head trainable
                if "history_network" not in name and "history_bias_head" not in name:
                    param.requires_grad_(False)
                    frozen_params.append((name, param))
            # Create dedicated optimizer for history modules only
            hb_lr = phase.history_bias_lr or trainer.config.learning_rate * 10
            history_params = list(trainer.model.win_history_network.parameters()) + \
                             list(trainer.model.lose_history_network.parameters()) + \
                             list(trainer.model.history_bias_head.parameters())
            trainer.optimizer = torch.optim.Adam(history_params, lr=hb_lr)

        # Train with phase-specific loss weights
        phase_metrics = trainer.train(loss_weights=phase.loss_weights)

        # Unfreeze and restore optimizer if we froze
        if original_optimizer is not None:
            for name, param in frozen_params:
                param.requires_grad_(True)
            trainer.optimizer = original_optimizer

        # Restore epoch count (commit window persists unless overridden later)
        trainer.config.epochs = original_epochs

        # Append phase metrics to cumulative tracking
        for key in phase_metrics:
            if key in cumulative_metrics:
                cumulative_metrics[key].extend(phase_metrics[key])
        
        # Save intermediate checkpoint if requested
        if curriculum.checkpoint_each_phase:
            phase_checkpoint_path = paths.root / f"model_{phase.name}.pt"
            torch.save(trainer.model.state_dict(), phase_checkpoint_path)
            print(f"Saved checkpoint: {phase_checkpoint_path}")
        
        # Evaluate phase success
        if phase.success_criteria and phase_idx < len(curriculum.phases) - 1:
            # Do a quick rollout to temp path for evaluation
            temp_paths = HybridDDMPaths(
                root=paths.root,
                config=paths.config,
                log=paths.root / f"trials_{phase.name}_eval.ndjson",
                metrics=paths.metrics,
                model=paths.model,
            )
            # Save original episodes, do quick eval rollout
            original_episodes = trainer.config.episodes
            trainer.config.episodes = 5
            _ = trainer.rollout(temp_paths)
            trainer.config.episodes = original_episodes
            
            success, eval_metrics = _evaluate_phase_success(
                temp_paths.log,
                phase.success_criteria,
                config.reference_log,
                task=config.task,
            )
            
            phase_result: dict = {
                "name": phase.name,
                "epochs": phase.epochs,
                "loss_weights": asdict(phase.loss_weights),
                "commit_window": {
                    "min": phase_min_commit,
                    "max": phase_max_commit,
                },
                "success": success,
                "metrics": eval_metrics,
                "criteria": phase.success_criteria,
            }
            phase_results.append(phase_result)
            
            print(f"\n{phase.name} Evaluation:")
            print(f"  Metrics: {eval_metrics}")
            print(f"  Criteria: {phase.success_criteria}")
            print(f"  Success: {'✓ PASSED' if success else '✗ FAILED'}\n")
            
            # Early stopping if criteria not met
            if not success and curriculum.allow_early_stopping:
                print(f"⚠️  Phase {phase.name} failed to meet success criteria.")
                print("   Stopping curriculum early. Consider:")
                print(f"   - Adjusting phase {phase.name} hyperparameters")
                print("   - Lowering success criteria")
                print("   - Trying supervised pretraining (Option C1)")
                break
        else:
            # Last phase or no criteria
            phase_result = {
                "name": phase.name,
                "epochs": phase.epochs,
                "loss_weights": asdict(phase.loss_weights),
                "commit_window": {
                    "min": phase_min_commit,
                    "max": phase_max_commit,
                },
                "success": True,  # Final phase always succeeds
                "metrics": {},
                "criteria": {},
            }
            phase_results.append(phase_result)
    
    # Final rollout with completed model
    print(f"\n{'='*80}")
    print("Running final rollout...")
    print(f"{'='*80}\n")
    rollout_stats = trainer.rollout(paths)
    
    # Save final state
    trainer.save(paths, cumulative_metrics, rollout_stats)
    
    # Save phase summary
    phase_summary_path = paths.root / "curriculum_phases.json"
    with open(phase_summary_path, "w") as f:
        json.dump({"phases": phase_results}, f, indent=2)
    print(f"\nSaved curriculum phase summary: {phase_summary_path}")
    
    # Return results
    return {
        "training_metrics": cumulative_metrics,
        "rollout_stats": rollout_stats,
        "phases": phase_results,
        "paths": {
            "log": str(paths.log),
            "config": str(paths.config),
            "metrics": str(paths.metrics),
            "model": str(paths.model),
        },
    }


def train_hybrid(config: HybridTrainingConfig) -> dict[str, Any]:
    """High-level entry point used by CLI/tests."""
    
    # Use curriculum training if configured
    if config.curriculum is not None:
        return train_hybrid_curriculum(config)
    
    trainer = HybridDDMTrainer(config)
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


