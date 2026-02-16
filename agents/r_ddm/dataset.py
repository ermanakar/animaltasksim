"""Dataset utilities for training the R-DDM agent on IBL 2AFC logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return default


@dataclass(slots=True)
class SessionSample:
    """Pre-computed arrays representing a single behavioural session."""

    session_id: str
    stimulus: np.ndarray
    block_prior: np.ndarray
    prev_action: np.ndarray  # -1 for none, else {0,1}
    prev_reward: np.ndarray
    prev_correct: np.ndarray
    action: np.ndarray
    correct: np.ndarray
    reward: np.ndarray
    rt_seconds: np.ndarray


class IBLRDDMDataset(Dataset[SessionSample]):
    """Dataset that groups IBL trials by session for sequence modelling."""

    def __init__(self, path: Path, *, max_sessions: int | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Reference log not found at {self.path}")
        self.sessions: list[SessionSample] = self._load_sessions(max_sessions=max_sessions)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> SessionSample:
        return self.sessions[idx]

    # ---------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_sessions(self, *, max_sessions: int | None) -> list[SessionSample]:
        grouped: dict[str, list[dict]] = {}
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                session_id = str(record.get("session_id", "unknown"))
                grouped.setdefault(session_id, []).append(record)

        session_ids = sorted(grouped)
        if max_sessions is not None:
            session_ids = session_ids[:max_sessions]

        sessions: list[SessionSample] = []
        for session_id in session_ids:
            trials = sorted(grouped[session_id], key=lambda t: t.get("trial_index", 0))
            sessions.append(self._build_session_sample(session_id, trials))
        return sessions

    def _build_session_sample(self, session_id: str, trials: Sequence[dict]) -> SessionSample:
        n = len(trials)

        stimulus = np.zeros(n, dtype=np.float32)
        block_prior = np.zeros(n, dtype=np.float32)
        action = np.zeros(n, dtype=np.int64)
        correct = np.zeros(n, dtype=np.float32)
        reward = np.zeros(n, dtype=np.float32)
        rt_seconds = np.zeros(n, dtype=np.float32)

        prev_action = np.full(n, -1, dtype=np.int64)
        prev_reward = np.zeros(n, dtype=np.float32)
        prev_correct = np.zeros(n, dtype=np.float32)

        for i, trial in enumerate(trials):
            stim = trial.get("stimulus", {})
            stimulus[i] = _safe_float(stim.get("contrast", 0.0))

            block = trial.get("block_prior", {})
            block_prior[i] = _safe_float(block.get("p_right", 0.5), 0.5)

            act = trial.get("action", 0)
            if isinstance(act, str):
                action[i] = 1 if act.lower() in {"1", "right"} else 0
            else:
                action[i] = int(act)

            reward_val = _safe_float(trial.get("reward", 0.0))
            reward[i] = reward_val
            correct[i] = 1.0 if bool(trial.get("correct", False)) else 0.0
            rt_seconds[i] = _safe_float(trial.get("rt_ms", 0.0)) / 1000.0

            prev = trial.get("prev")
            if isinstance(prev, dict):
                prev_act = prev.get("action", -1)
                if isinstance(prev_act, str):
                    prev_action[i] = 0 if prev_act.lower() in {"left", "0"} else 1 if prev_act.lower() in {"right", "1"} else -1
                else:
                    prev_action[i] = int(prev_act)
                prev_reward[i] = _safe_float(prev.get("reward", 0.0))
                prev_correct[i] = 1.0 if bool(prev.get("correct", False)) else 0.0

        return SessionSample(
            session_id=session_id,
            stimulus=stimulus,
            block_prior=block_prior,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_correct=prev_correct,
            action=action,
            correct=correct,
            reward=reward,
            rt_seconds=rt_seconds,
        )


def rddm_collate_sessions(batch: Sequence[SessionSample]) -> dict[str, torch.Tensor]:
    """Collate function that pads variable-length session sequences."""

    lengths = torch.tensor([len(sample.action) for sample in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)

    def _pad(arrays: Iterable[np.ndarray], *, dtype: torch.dtype, fill: float = 0.0) -> torch.Tensor:
        output = torch.full((batch_size, max_len), fill, dtype=dtype)
        for row, array in enumerate(arrays):
            length = len(array)
            output[row, :length] = torch.as_tensor(array, dtype=dtype)
        return output

    stimulus = _pad((s.stimulus for s in batch), dtype=torch.float32)
    block_prior = _pad((s.block_prior for s in batch), dtype=torch.float32)
    actions = _pad((s.action for s in batch), dtype=torch.long)
    correct = _pad((s.correct for s in batch), dtype=torch.float32)
    reward = _pad((s.reward for s in batch), dtype=torch.float32)
    rt_seconds = _pad((s.rt_seconds for s in batch), dtype=torch.float32)

    prev_action = _pad((s.prev_action for s in batch), dtype=torch.long, fill=-1)
    prev_reward = _pad((s.prev_reward for s in batch), dtype=torch.float32)
    prev_correct = _pad((s.prev_correct for s in batch), dtype=torch.float32)

    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "stimulus": stimulus,
        "block_prior": block_prior,
        "actions": actions,
        "correct": correct,
        "reward": reward,
        "rt_seconds": rt_seconds,
        "prev_action": prev_action,
        "prev_reward": prev_reward,
        "prev_correct": prev_correct,
        "mask": mask,
        "lengths": lengths,
    }


class RDMRDDMDataset(IBLRDDMDataset):
    """Dataset for macaque random-dot motion sessions."""

    def _build_session_sample(self, session_id: str, trials: Sequence[dict]) -> SessionSample:
        n = len(trials)

        stimulus = np.zeros(n, dtype=np.float32)
        block_prior = np.full(n, 0.5, dtype=np.float32)
        action = np.zeros(n, dtype=np.int64)
        correct = np.zeros(n, dtype=np.float32)
        reward = np.zeros(n, dtype=np.float32)
        rt_seconds = np.zeros(n, dtype=np.float32)

        prev_action = np.full(n, -1, dtype=np.int64)
        prev_reward = np.zeros(n, dtype=np.float32)
        prev_correct = np.zeros(n, dtype=np.float32)

        for i, trial in enumerate(trials):
            stim = trial.get("stimulus", {})
            stimulus[i] = _safe_float(stim.get("coherence", 0.0))

            act = trial.get("action", 0)
            if isinstance(act, str):
                act_lower = act.lower()
                if act_lower in {"right", "1"}:
                    action[i] = 1
                elif act_lower in {"left", "0"}:
                    action[i] = 0
                else:
                    action[i] = 1 if stimulus[i] >= 0 else 0
            else:
                action[i] = int(act)

            reward_val = _safe_float(trial.get("reward", 0.0))
            reward[i] = reward_val
            correct[i] = 1.0 if bool(trial.get("correct", False)) else 0.0
            rt_seconds[i] = _safe_float(trial.get("rt_ms", 0.0)) / 1000.0

            prev = trial.get("prev")
            if isinstance(prev, dict):
                prev_act = prev.get("action", -1)
                if isinstance(prev_act, str):
                    prev_lower = prev_act.lower()
                    if prev_lower in {"right", "1"}:
                        prev_action[i] = 1
                    elif prev_lower in {"left", "0"}:
                        prev_action[i] = 0
                    else:
                        prev_action[i] = -1
                else:
                    prev_action[i] = int(prev_act)
                prev_reward[i] = _safe_float(prev.get("reward", 0.0))
                prev_correct[i] = 1.0 if bool(prev.get("correct", False)) else 0.0

        return SessionSample(
            session_id=session_id,
            stimulus=stimulus,
            block_prior=block_prior,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_correct=prev_correct,
            action=action,
            correct=correct,
            reward=reward,
            rt_seconds=rt_seconds,
        )
