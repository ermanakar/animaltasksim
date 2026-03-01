#!/usr/bin/env python
"""Generate publication-quality behavioral fingerprint figures for the README."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from eval.metrics import (
    compute_history_metrics,
    load_trials,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLOR_AGENT = "#2166ac"
COLOR_REF = "#4d4d4d"
MARKER_AGENT = "o"
MARKER_REF = "s"


@dataclass(slots=True)
class FigureConfig:
    """Configuration for README figure generation."""

    agent_logs: tuple[str, ...] = ()
    """Paths to agent trial logs (one per seed)."""

    reference_log: str = "data/ibl/reference.ndjson"
    """Path to reference animal data."""

    output_dir: str = "docs/figures"
    """Directory for output PNGs."""

    hero_only: bool = False
    """Only generate the combined 3-panel hero figure, skip individual panels."""

    dpi: int = 300
    """Output resolution."""

    def __post_init__(self) -> None:
        if not self.agent_logs:
            # Default: contrast_fix_5seed runs
            seeds = [42, 123, 456, 789, 1337]
            self.agent_logs = [
                f"runs/contrast_fix_5seed/seed{s}/trials.ndjson" for s in seeds
            ]


# ---------------------------------------------------------------------------
# Psychometric data helpers
# ---------------------------------------------------------------------------

def _logistic(x: np.ndarray, bias: float, slope: float, gl: float, gh: float) -> np.ndarray:
    return gl + (1.0 - gl - gh) / (1.0 + np.exp(-slope * (x - bias)))


def _psychometric_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (signed_contrasts, p_right) grouped arrays."""
    filtered = df[df["stimulus_contrast"].notnull()].copy()
    filtered["choice_right"] = (filtered["action"] == "right").astype(float)
    grouped = filtered.groupby("stimulus_contrast")["choice_right"].mean().sort_index()
    return grouped.index.values.astype(float), grouped.values.astype(float)


def _chronometric_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (|contrast|, median_rt_ms) grouped arrays."""
    data = df.copy()
    data["rt_used"] = data["rt_ms"].fillna(0.0)
    data = data[data["rt_used"] > 0.0]
    data["difficulty"] = np.abs(data["stimulus_contrast"])
    grouped = data.groupby("difficulty")["rt_ms"].median().sort_index()
    return grouped.index.values.astype(float), grouped.values.astype(float)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_psychometric(
    ax: plt.Axes,
    agent_dfs: list[pd.DataFrame],
    ref_df: pd.DataFrame,
) -> None:
    """Panel (a): psychometric curves."""
    # Reference
    ref_x, ref_y = _psychometric_curve(ref_df)
    ax.plot(ref_x, ref_y, marker=MARKER_REF, color=COLOR_REF, markerfacecolor="none",
            linewidth=1.5, markersize=7, linestyle="--", label="IBL mouse", zorder=2)

    # Agent: collect per-seed curves, interpolate to common x, compute mean ± SEM
    all_x = set()
    seed_curves = []
    for adf in agent_dfs:
        sx, sy = _psychometric_curve(adf)
        all_x.update(sx.tolist())
        seed_curves.append((sx, sy))

    common_x = np.array(sorted(all_x))
    interp_y = np.full((len(agent_dfs), len(common_x)), np.nan)
    for i, (sx, sy) in enumerate(seed_curves):
        for j, cx in enumerate(common_x):
            idx = np.where(np.isclose(sx, cx))[0]
            if len(idx) > 0:
                interp_y[i, j] = sy[idx[0]]

    mean_y = np.nanmean(interp_y, axis=0)
    sem_y = np.nanstd(interp_y, axis=0) / np.sqrt(np.sum(~np.isnan(interp_y), axis=0))

    ax.plot(common_x, mean_y, marker=MARKER_AGENT, color=COLOR_AGENT,
            linewidth=2, markersize=7, label="Hybrid DDM+LSTM", zorder=3)
    ax.fill_between(common_x, mean_y - sem_y, mean_y + sem_y,
                     color=COLOR_AGENT, alpha=0.2, zorder=1)

    ax.set_xlabel("Signed contrast")
    ax.set_ylabel("P(rightward choice)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("(a) Psychometric curve", fontweight="bold", fontsize=11)


def plot_chronometric(
    ax: plt.Axes,
    agent_dfs: list[pd.DataFrame],
    ref_df: pd.DataFrame,
) -> None:
    """Panel (b): chronometric curves."""
    # Reference
    ref_x, ref_y = _chronometric_curve(ref_df)
    ax.plot(ref_x, ref_y, marker=MARKER_REF, color=COLOR_REF, markerfacecolor="none",
            linewidth=1.5, markersize=7, linestyle="--", label="IBL mouse", zorder=2)

    # Agent: per-seed curves
    all_x = set()
    seed_curves = []
    for adf in agent_dfs:
        sx, sy = _chronometric_curve(adf)
        all_x.update(sx.tolist())
        seed_curves.append((sx, sy))

    common_x = np.array(sorted(all_x))
    interp_y = np.full((len(agent_dfs), len(common_x)), np.nan)
    for i, (sx, sy) in enumerate(seed_curves):
        for j, cx in enumerate(common_x):
            idx = np.where(np.isclose(sx, cx))[0]
            if len(idx) > 0:
                interp_y[i, j] = sy[idx[0]]

    mean_y = np.nanmean(interp_y, axis=0)
    sem_y = np.nanstd(interp_y, axis=0) / np.sqrt(np.sum(~np.isnan(interp_y), axis=0))

    ax.plot(common_x, mean_y, marker=MARKER_AGENT, color=COLOR_AGENT,
            linewidth=2, markersize=7, label="Hybrid DDM+LSTM", zorder=3)
    ax.fill_between(common_x, mean_y - sem_y, mean_y + sem_y,
                     color=COLOR_AGENT, alpha=0.2, zorder=1)

    ax.set_xlabel("|Contrast|")
    ax.set_ylabel("Median RT (ms)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("(b) Chronometric curve", fontweight="bold", fontsize=11)


def plot_history(
    ax: plt.Axes,
    agent_dfs: list[pd.DataFrame],
    ref_df: pd.DataFrame,
) -> None:
    """Panel (c): win-stay and lose-shift bar chart."""
    # Reference
    ref_hist = compute_history_metrics(ref_df)

    # Agent: per-seed metrics
    ws_vals = []
    ls_vals = []
    for adf in agent_dfs:
        h = compute_history_metrics(adf)
        ws_vals.append(h.win_stay)
        ls_vals.append(h.lose_shift)

    ws_mean, ws_sem = np.mean(ws_vals), np.std(ws_vals) / np.sqrt(len(ws_vals))
    ls_mean, ls_sem = np.mean(ls_vals), np.std(ls_vals) / np.sqrt(len(ls_vals))

    x = np.array([0, 1])
    width = 0.3

    # Reference bars
    ax.bar(x - width / 2, [ref_hist.win_stay, ref_hist.lose_shift],
           width, color=COLOR_REF, alpha=0.5, edgecolor=COLOR_REF,
           label="IBL mouse", zorder=2)

    # Agent bars
    ax.bar(x + width / 2, [ws_mean, ls_mean], width,
           yerr=[ws_sem, ls_sem], capsize=4,
           color=COLOR_AGENT, alpha=0.7, edgecolor=COLOR_AGENT,
           label="Hybrid DDM+LSTM", zorder=3)

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Win-stay", "Lose-shift"])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("(c) History effects", fontweight="bold", fontsize=11)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = tyro.cli(FigureConfig)

    # Load data
    print(f"Loading reference: {cfg.reference_log}")
    ref_df = load_trials(cfg.reference_log)

    agent_dfs: list[pd.DataFrame] = []
    for log_path in cfg.agent_logs:
        p = Path(log_path)
        if not p.exists():
            print(f"  WARNING: {log_path} not found, skipping")
            continue
        print(f"  Loading agent seed: {log_path}")
        agent_dfs.append(load_trials(log_path))

    if not agent_dfs:
        print("ERROR: No agent data found. Provide --agent-logs or run contrast_fix_5seed first.")
        raise SystemExit(1)

    print(f"Loaded {len(agent_dfs)} agent seeds, {len(ref_df)} reference trials")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Hero figure (3-panel) ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("white")

    plot_psychometric(axes[0], agent_dfs, ref_df)
    plot_chronometric(axes[1], agent_dfs, ref_df)
    plot_history(axes[2], agent_dfs, ref_df)

    fig.tight_layout(pad=2.0)
    hero_path = out_dir / "behavioral_fingerprint.png"
    fig.savefig(hero_path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved hero figure: {hero_path}")

    # --- Individual panels ---
    if not cfg.hero_only:
        for panel_name, plot_fn in [
            ("psychometric", plot_psychometric),
            ("chronometric", plot_chronometric),
            ("history", plot_history),
        ]:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(5, 4))
            fig_single.patch.set_facecolor("white")
            plot_fn(ax_single, agent_dfs, ref_df)
            fig_single.tight_layout()
            panel_path = out_dir / f"{panel_name}.png"
            fig_single.savefig(panel_path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig_single)
            print(f"Saved panel: {panel_path}")

    print("Done!")


if __name__ == "__main__":
    main()
