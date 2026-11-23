"""The following-trial control must cancel slow side preference and keep a planted outcome effect."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.explore_deep_reference import fit_history


def _session(win_stay: float, seed: int = 3, trials: int = 60000) -> pd.DataFrame:
    """One long session with slowly drifting side preference and an optional reward-repeat effect."""
    rng = np.random.default_rng(seed)
    signed = rng.choice([-1.0, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 1.0], trials)
    choice, reward = np.zeros(trials), np.zeros(trials)
    drift = 0.0
    for t in range(trials):
        drift = 0.995 * drift + rng.normal(0, 0.12)
        score = 8 * signed[t] + drift
        if t and reward[t - 1]:
            score += win_stay * choice[t - 1]
        choice[t] = 1.0 if rng.random() < expit(score) else -1.0
        reward[t] = float(choice[t] == np.sign(signed[t]))
    frame = pd.DataFrame({"signed": signed, "pl": 0.5, "right": (choice == 1).astype(float)})
    for name, shift in (("prev", 1), ("next", -1)):
        frame[f"{name}_choice"] = pd.Series(choice).shift(shift)
        frame[f"{name}_reward"] = pd.Series(reward).shift(shift)
        frame[f"{name}_strength"] = pd.Series(np.abs(signed)).shift(shift)
    return frame


def _net(frame: pd.DataFrame, name: str) -> tuple[float, float]:
    past, future = fit_history(frame, "prev"), fit_history(frame, "next")
    assert past is not None and future is not None
    return past[name], past[name] - future[name]


def test_following_trial_control_separates_drift_from_outcome_history() -> None:
    raw, net = _net(_session(win_stay=0.0), "reward_hard")
    assert raw > 0.2  # drift alone looks like repetition after reward
    assert abs(net) < 0.1
    _, planted = _net(_session(win_stay=0.6), "reward_hard")
    assert planted > 0.3
