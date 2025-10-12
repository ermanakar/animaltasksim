"""Hybrid DDM + LSTM agent designed to mimic animal behaviour on RDM."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.losses import (
    LossWeights,
    choice_loss,
    history_penalty,
    non_decision_supervision_loss,
    rt_loss,
)
from agents.wfpt_loss import wfpt_loss
from animaltasksim.config import ProjectPaths
from animaltasksim.seeding import seed_everything
from envs.rdm_macaque import (
    ACTION_HOLD,
    ACTION_LEFT,
    ACTION_RIGHT,
    AgentMetadata,
    RDMConfig,
    RDMMacaqueEnv,
)
from eval.metrics import load_trials


@dataclass(slots=True)
class HybridDDMPaths:
    """Collection of output artefact locations."""

    root: Path
    config: Path
    log: Path
    metrics: Path
    model: Path


@dataclass(slots=True)
class CurriculumPhase:
    """Configuration for a single curriculum learning phase."""

    name: str
    epochs: int
    loss_weights: LossWeights
    success_criteria: dict[str, float] = field(default_factory=dict)
    # Success criteria keys: 'min_slope_abs', 'min_r2', 'min_rt_diff_abs'


@dataclass(slots=True)
class CurriculumConfig:
    """Multi-phase curriculum learning schedule."""

    phases: List[CurriculumPhase] = field(default_factory=list)
    allow_early_stopping: bool = True
    checkpoint_each_phase: bool = True

    @staticmethod
    def default_rt_first() -> CurriculumConfig:
        """Default 2-phase curriculum: WFPT → full balance."""
        phase1 = CurriculumPhase(
            name="phase1_wfpt_only",
            epochs=10,
            loss_weights=LossWeights(
                choice=0.0,
                rt=0.0,
                history=0.0,
                drift_supervision=0.1,
                non_decision_supervision=0.1,
                wfpt=1.0,
            ),
            success_criteria={"min_slope_abs": 100.0, "min_r2": 0.1},
        )
        phase2 = CurriculumPhase(
            name="phase2_full_balance",
            epochs=5,
            loss_weights=LossWeights(
                choice=1.0,
                rt=0.0,
                history=0.1,
                drift_supervision=0.05,
                non_decision_supervision=0.05,
                wfpt=0.5,
            ),
            success_criteria={},
        )
        return CurriculumConfig(phases=[phase1, phase2])


@dataclass(slots=True)
class HybridTrainingConfig:
    """Training and rollout configuration for the hybrid agent."""

    reference_log: Path = ProjectPaths.from_cwd().data / "macaque" / "reference.ndjson"
    output_dir: Path = field(default_factory=lambda: ProjectPaths.from_cwd().runs / "rdm_hybrid")
    agent_version: str = "0.1.0"
    trials_per_episode: int = 400
    episodes: int = 10
    seed: int = 1234
    epochs: int = 5
    hidden_size: int = 64
    learning_rate: float = 1e-3
    loss_weights: LossWeights = field(default_factory=LossWeights)
    step_ms: int = 10
    max_sessions: int | None = None
    max_trials_per_session: int | None = None
    min_commit_steps: int = 5
    max_commit_steps: int = 120
    drift_scale: float = 10.0  # Scale drift_head initialization to enable stronger evidence effects
    curriculum: CurriculumConfig | None = None  # If set, use curriculum learning

    def output_paths(self) -> HybridDDMPaths:
        out = Path(self.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return HybridDDMPaths(
            root=out,
            config=out / "config.json",
            log=out / "trials.ndjson",
            metrics=out / "training_metrics.json",
            model=out / "model.pt",
        )


@dataclass(slots=True)
class SessionBatch:
    """Reference data for a single animal session."""

    features: np.ndarray  # shape (T, F)
    choice: np.ndarray  # shape (T,)
    choice_mask: np.ndarray  # shape (T,)
    rt_ms: np.ndarray  # shape (T,)
    rt_mask: np.ndarray  # shape (T,)
    correct: np.ndarray  # shape (T,)
    win_stay_target: float
    lose_shift_target: float


class HybridDDMModel(nn.Module):
    """Controller that maps history-aware features to DDM parameters."""

    def __init__(self, feature_dim: int, hidden_size: int, device: torch.device, drift_scale: float = 10.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.LSTMCell(feature_dim, hidden_size)
        # DDM parameter heads
        self.drift_head = nn.Linear(hidden_size, 1)
        self.bound_head = nn.Linear(hidden_size, 1)
        self.bias_head = nn.Linear(hidden_size, 1)
        self.non_decision_head = nn.Linear(hidden_size, 1)
        
        # CRITICAL CALIBRATION: Initialize drift_head bias for realistic drift_gain scale
        # Target: drift_gain ~ 10-15 to match macaque RT dynamics (RT ~ 500-800ms)
        # Formula: drift_gain = softplus(bias) + 1e-3
        # For drift_gain ≈ 12: need bias ≈ 2.5
        nn.init.constant_(self.drift_head.bias, 2.5)
        self.log_noise = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self._min_bound = 0.5
        self._min_non_decision = 150.0  # ms
        
        # Initialize drift_head with stronger weights to enable evidence-dependent RTs
        with torch.no_grad():
            self.drift_head.weight.data *= drift_scale
            self.drift_head.bias.data *= drift_scale

    def init_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state
        h, c = self.lstm(x, (h_prev, c_prev))
        drift_gain = F.softplus(self.drift_head(h)) + 1e-3
        bound = F.softplus(self.bound_head(h)) + self._min_bound
        bias = torch.tanh(self.bias_head(h))
        non_decision = F.softplus(self.non_decision_head(h)) + self._min_non_decision
        safe_log_noise = torch.nan_to_num(self.log_noise, nan=0.0, posinf=5.0, neginf=-5.0)
        safe_log_noise = torch.clamp(safe_log_noise, -5.0, 5.0)
        noise = torch.exp(safe_log_noise) + 1e-3
        outputs = {
            "drift_gain": drift_gain.squeeze(-1),
            "bound": bound.squeeze(-1),
            "bias": bias.squeeze(-1),
            "non_decision_ms": non_decision.squeeze(-1),
            "noise": noise.squeeze(-1).expand_as(drift_gain.squeeze(-1)),
        }
        return outputs, (h, c)


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
            drift_scale=self.config.drift_scale
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
        df = df[df["task"] == "rdm"].copy()
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
            coherence = float(row.get("stimulus_coherence", 0.0))
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
            "epoch_history_penalty": [],
            "epoch_drift_supervision": [],
            "epoch_non_decision_supervision": [],
            "epoch_drift_magnitude": [],
            "epoch_wfpt_loss": [],
            "noise": [],
            "mean_bound": [],
        }
        for epoch in range(self.config.epochs):
            epoch_choice = 0.0
            epoch_rt = 0.0
            epoch_hist = 0.0
            epoch_drift_sup = 0.0
            epoch_non_decision_sup = 0.0
            epoch_drift_mag = 0.0
            epoch_wfpt = 0.0
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
                drift_gain_buffer: list[torch.Tensor] = []  # Collect for supervision
                
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

                    score = 2.0 * drift * bound / (noise**2)
                    score = torch.nan_to_num(score, nan=0.0, posinf=10.0, neginf=-10.0)
                    score = torch.clamp(score, -10.0, 10.0)
                    prob_right = torch.sigmoid(score)
                    prob_right = torch.nan_to_num(prob_right, nan=0.5)
                    prob_buffer.append(float(prob_right.detach().cpu()))

                    abs_drift = torch.abs(drift) + 1e-3
                    kappa = abs_drift * bound / (noise**2)
                    mean_steps = torch.where(
                        abs_drift > 1e-3,
                        bound / abs_drift * torch.tanh(kappa),
                        (bound**2) / (noise**2 + 1e-3),
                    )
                    predicted_rt = out["non_decision_ms"] + mean_steps * self.config.step_ms
                    predicted_rt = torch.nan_to_num(
                        predicted_rt,
                        nan=float(self.config.step_ms * self.config.min_commit_steps),
                        posinf=float(self.config.step_ms * self.config.max_commit_steps),
                        neginf=float(self.config.step_ms * self.config.min_commit_steps),
                    )
                    predicted_rt = torch.clamp(
                        predicted_rt,
                        min=float(self.config.step_ms * self.config.min_commit_steps),
                        max=float(self.config.step_ms * self.config.max_commit_steps),
                    )

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
                        wfpt_drift_buffer.append(drift.unsqueeze(0))
                        wfpt_bound_buffer.append(bound.unsqueeze(0))
                        wfpt_bias_buffer.append(out["bias"].unsqueeze(0))
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
                    # Target: drift_gain ≈ 12 (from softplus(2.5) + 0.001)
                    drift_mag_loss = torch.mean((drift_gains - 12.0) ** 2)
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
                epoch_hist += float(hist_loss.detach().cpu())
                epoch_drift_sup += float(drift_sup_loss.detach().cpu())
                epoch_non_decision_sup += float(non_decision_sup_loss.detach().cpu())
                epoch_drift_mag += float(drift_mag_loss.detach().cpu())
                epoch_wfpt += float(wfpt_loss_val.detach().cpu())
                session_count += 1

            metrics["epoch_choice_loss"].append(epoch_choice / max(session_count, 1))
            metrics["epoch_rt_loss"].append(epoch_rt / max(session_count, 1))
            metrics["epoch_history_penalty"].append(epoch_hist / max(session_count, 1))
            metrics["epoch_drift_supervision"].append(epoch_drift_sup / max(session_count, 1))
            metrics["epoch_non_decision_supervision"].append(
                epoch_non_decision_sup / max(session_count, 1)
            )
            metrics["epoch_drift_magnitude"].append(epoch_drift_mag / max(session_count, 1))
            metrics["epoch_wfpt_loss"].append(epoch_wfpt / max(session_count, 1))
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
        env_config = RDMConfig(
            trials_per_episode=self.config.trials_per_episode,
            log_path=paths.log,
            agent=AgentMetadata(name="hybrid_ddm", version=self.config.agent_version),
            seed=self.config.seed,
            per_step_cost=0.01,
            evidence_gain=0.05,
            momentary_sigma=1.0,
            collapsing_bound=False,  # CRITICAL FIX: Disable env's auto-commit so agent's DDM controls timing!
            min_bound_steps=self.config.min_commit_steps,  # Allow model's RT predictions through
        )
        env = RDMMacaqueEnv(env_config)
        step_ms = env.config.step_ms
        metrics: dict[str, list[float]] = {
            "cumulative_reward": [],
            "mean_rt_ms": [],
        }
        for episode in range(self.config.episodes):
            observation, info = env.reset(seed=self.config.seed + episode)
            h, c = self.model.init_state()
            cumulative_reward = 0.0
            planned_action = ACTION_HOLD
            commit_step_target = env.config.min_bound_steps
            rt_tracker: list[float] = []
            prev_action_val = 0.0
            prev_reward = 0.0
            prev_correct = 0.0
            current_coherence = float(getattr(env, "_signed_coherence", 0.0))
            while True:
                phase = info["phase"]
                if phase == "response":
                    if info["phase_step"] == 0:
                        current_coherence = float(getattr(env, "_signed_coherence", 0.0))
                        trial_idx_val = info.get("trial_index", 0)
                        # Type narrowing for info dict values
                        trial_idx = int(trial_idx_val) if isinstance(trial_idx_val, (int, float)) else 0
                        trial_norm = float(trial_idx) / max(
                            self.config.trials_per_episode, 1
                        )
                        features = self._features_from_trial(
                            current_coherence,
                            prev_action_val,
                            prev_reward,
                            prev_correct,
                            trial_norm,
                        )
                        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
                        out, (h, c) = self.model(x, (h, c))
                        
                        # Extract DDM parameters
                        coherence = x[0, 0].item()
                        drift_gain = out["drift_gain"].item()
                        bound = out["bound"].item()
                        noise = out["noise"].item()
                        bias = out["bias"].item()
                        non_decision_ms = out["non_decision_ms"].item()
                        
                        # Compute drift from coherence and drift_gain
                        drift = drift_gain * coherence
                        
                        # Run stochastic DDM simulation
                        # dt corresponds to step_ms in seconds
                        dt = self.config.step_ms / 1000.0  # Convert to seconds
                        planned_action, ddm_steps = self._simulate_ddm(
                            drift=drift,
                            bound=bound,
                            noise=noise,
                            bias=bias,
                            dt=dt,
                            max_steps=self.config.max_commit_steps,
                        )
                        
                        # Calculate commit time: non-decision delay + DDM accumulation time
                        ddm_time_ms = ddm_steps * self.config.step_ms
                        total_rt_ms = non_decision_ms + ddm_time_ms
                        
                        # Ensure within bounds
                        total_rt_ms = np.clip(
                            total_rt_ms,
                            self.config.step_ms * self.config.min_commit_steps,
                            self.config.step_ms * self.config.max_commit_steps,
                        )
                        
                        commit_step_target = int(total_rt_ms / step_ms)
                        commit_step_target = np.clip(
                            commit_step_target,
                            self.config.min_commit_steps,
                            self.config.max_commit_steps,
                        )
                    # Type-safe comparison with phase_step
                    phase_step_val = info.get("phase_step", 0)
                    current_step = int(phase_step_val) if isinstance(phase_step_val, (int, float)) else 0
                    if current_step + 1 >= commit_step_target:
                        action = planned_action
                    else:
                        action = ACTION_HOLD
                else:
                    action = ACTION_HOLD
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
                    expected_right = current_coherence >= 0.0
                    prev_correct = 1.0 if (planned_action == ACTION_RIGHT) == expected_right else 0.0
                    prev_reward = float(reward)
                if info["phase"] == "fixation" and info.get("phase_step", 0) == 0:
                    # New trial about to begin; keep previous summary from last outcome.
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
) -> tuple[bool, dict[str, float]]:
    """Evaluate whether phase success criteria are met."""
    from scipy import stats as scipy_stats
    
    # Load agent trials
    trials: list[dict] = []
    with open(trials_path) as f:
        for line in f:
            trials.append(json.loads(line))
    
    if not trials:
        return False, {}
    
    rts = np.array([t['rt_ms'] for t in trials])
    coherences = np.array([t['stimulus']['coherence'] for t in trials])
    abs_coh = np.abs(coherences)
    
    # Compute RT-coherence metrics
    linreg_result = scipy_stats.linregress(abs_coh, rts)
    # scipy.stats.linregress returns LinregressResult with .slope and .rvalue attributes
    slope_val = float(linreg_result.slope)  # type: ignore[attr-defined]
    r_val = float(linreg_result.rvalue)  # type: ignore[attr-defined]
    r2_val = r_val ** 2
    rt_diff = float(np.abs(rts[abs_coh < 0.05].mean() - rts[abs_coh > 0.5].mean()))
    
    metrics = {
        "slope_abs": abs(slope_val),
        "r2": r2_val,
        "rt_diff_abs": rt_diff,
    }
    
    # Check criteria
    success = True
    if "min_slope_abs" in criteria and metrics["slope_abs"] < criteria["min_slope_abs"]:
        success = False
    if "min_r2" in criteria and metrics["r2"] < criteria["min_r2"]:
        success = False
    if "min_rt_diff_abs" in criteria and metrics["rt_diff_abs"] < criteria["min_rt_diff_abs"]:
        success = False
    
    return success, metrics


def train_hybrid_curriculum(config: HybridTrainingConfig) -> dict:
    """Train using curriculum learning with phased loss weights."""
    if config.curriculum is None:
        raise ValueError("Curriculum config required for curriculum training")
    
    curriculum = config.curriculum
    trainer = HybridDDMTrainer(config)
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
        print(f"{'='*80}\n")
        
        # Temporarily update config for this phase
        original_epochs = trainer.config.epochs
        trainer.config.epochs = phase.epochs
        
        # Train with phase-specific loss weights
        phase_metrics = trainer.train(loss_weights=phase.loss_weights)
        
        # Restore original epochs
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
            )
            
            phase_result: dict = {
                "name": phase.name,
                "epochs": phase.epochs,
                "loss_weights": asdict(phase.loss_weights),
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
                print(f"   Stopping curriculum early. Consider:")
                print(f"   - Adjusting phase {phase.name} hyperparameters")
                print(f"   - Lowering success criteria")
                print(f"   - Trying supervised pretraining (Option C1)")
                break
        else:
            # Last phase or no criteria
            phase_result = {
                "name": phase.name,
                "epochs": phase.epochs,
                "loss_weights": asdict(phase.loss_weights),
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


def train_hybrid(config: HybridTrainingConfig) -> dict[str, object]:
    """High-level entry point used by CLI/tests."""
    
    # Use curriculum training if configured
    if config.curriculum is not None:
        return train_hybrid_curriculum(config)
    
    trainer = HybridDDMTrainer(config)
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


__all__ = ["HybridTrainingConfig", "HybridDDMTrainer", "train_hybrid", "train_hybrid_curriculum", "LossWeights", "CurriculumConfig", "CurriculumPhase"]
