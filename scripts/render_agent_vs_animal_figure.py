#!/usr/bin/env python
"""Render the README's Figure 1: adaptive-control agent vs. IBL mouse.

Three panels (psychometric, chronometric, history) styled for high-impact
journal layouts: sans-serif typography, no top/right spines, no in-axes
panel titles (those live in the README caption), bold a/b/c labels in
the top-left of each panel, and a two-tone agent-vs-reference palette.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from _figure_style import (  # noqa: E402
    COLOR_AGENT,
    COLOR_REFERENCE,
    add_panel_label,
    apply_journal_style,
)
from eval.metrics import compute_all_metrics, load_trials


def _stim_column(df: pd.DataFrame) -> tuple[str, str]:
    if "stimulus_contrast" in df.columns:
        return "stimulus_contrast", "Stimulus contrast"
    if "stimulus_coherence" in df.columns:
        return "stimulus_coherence", "Stimulus coherence"
    raise SystemExit("No stimulus column found.")


def _psychometric_curve(x: np.ndarray, params: dict) -> np.ndarray:
    slope = params["slope"]
    bias = params.get("bias", 0.0)
    lapse_low = params.get("lapse_low", 0.0)
    lapse_high = params.get("lapse_high", 0.0)
    core = 1.0 / (1.0 + np.exp(-(x - bias) * slope))
    return lapse_low + (1.0 - lapse_low - lapse_high) * core


def _plot_psychometric(ax, df_agent, df_animal, m_agent, m_animal, stim_col):
    by_agent = df_agent.groupby(stim_col)["action"].apply(lambda s: (s == "right").mean())
    by_animal = df_animal.groupby(stim_col)["action"].apply(lambda s: (s == "right").mean())

    ax.scatter(
        by_animal.index, by_animal.values,
        s=42, facecolors="none", edgecolors=COLOR_REFERENCE,
        linewidths=1.2, zorder=4, label="IBL mouse",
    )
    ax.scatter(
        by_agent.index, by_agent.values,
        s=42, color=COLOR_AGENT, zorder=5, label="Agent",
    )

    x_lo = float(min(by_agent.index.min(), by_animal.index.min()))
    x_hi = float(max(by_agent.index.max(), by_animal.index.max()))
    xs = np.linspace(x_lo, x_hi, 200)

    p_animal = m_animal.get("psychometric", {})
    if p_animal and np.isfinite(p_animal.get("slope", np.nan)):
        ax.plot(xs, _psychometric_curve(xs, p_animal),
                color=COLOR_REFERENCE, linewidth=1.5, linestyle="--", alpha=0.85)
    p_agent = m_agent.get("psychometric", {})
    if p_agent and np.isfinite(p_agent.get("slope", np.nan)):
        ax.plot(xs, _psychometric_curve(xs, p_agent),
                color=COLOR_AGENT, linewidth=1.8)

    ax.set_xlabel("Stimulus contrast")
    ax.set_ylabel("P(rightward choice)")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(loc="upper left", handletextpad=0.5)


def _plot_chronometric(ax, df_agent, df_animal, stim_col):
    def _by_difficulty(df):
        d = df.copy()
        d["abs_stim"] = d[stim_col].abs()
        grouped = d.groupby("abs_stim")["rt_ms"]
        out = grouped.agg(["median", "count"])
        # SE of the median ≈ 1.2533 * std / sqrt(n).
        out["sem"] = grouped.std().fillna(0) * 1.2533 / np.sqrt(out["count"])
        return out.reset_index()

    a = _by_difficulty(df_agent)
    r = _by_difficulty(df_animal)

    ax.errorbar(
        r["abs_stim"], r["median"], yerr=r["sem"],
        fmt="s", mfc="none", mec=COLOR_REFERENCE,
        ecolor=COLOR_REFERENCE, elinewidth=0.9, capsize=2.2,
        markersize=5, linestyle="--", linewidth=1.2,
        color=COLOR_REFERENCE, label="IBL mouse",
    )
    ax.errorbar(
        a["abs_stim"], a["median"], yerr=a["sem"],
        fmt="o", color=COLOR_AGENT, ecolor=COLOR_AGENT,
        elinewidth=0.9, capsize=2.2, markersize=5,
        linewidth=1.6, label="Agent",
    )

    ax.set_xlabel("|Stimulus contrast|")
    ax.set_ylabel("Median reaction time (ms)")
    ax.legend(loc="upper right")


def _plot_history(ax, m_agent, m_animal):
    metrics = [
        ("win_stay", "Win-stay"),
        ("lose_shift", "Lose-shift"),
        ("sticky_choice", "Sticky-choice"),
    ]
    h_agent = m_agent.get("history", {})
    h_animal = m_animal.get("history", {})

    xs = np.arange(len(metrics))
    width = 0.36

    agent_vals = [float(h_agent.get(k, np.nan)) for k, _ in metrics]
    animal_vals = [float(h_animal.get(k, np.nan)) for k, _ in metrics]

    ax.bar(
        xs - width / 2, animal_vals,
        width=width, color="white",
        edgecolor=COLOR_REFERENCE, linewidth=1.2,
        label="IBL mouse",
    )
    ax.bar(
        xs + width / 2, agent_vals,
        width=width, color=COLOR_AGENT,
        edgecolor=COLOR_AGENT, linewidth=1.0,
        label="Agent",
    )

    for x, v in zip(xs - width / 2, animal_vals):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.0, color=COLOR_REFERENCE)
    for x, v in zip(xs + width / 2, agent_vals):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.0, color=COLOR_AGENT)

    ax.axhline(0.5, color="#999999", linewidth=0.7, linestyle=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right")


@dataclass(slots=True)
class Args:
    agent_log: Path = Path(
        "runs/adaptive_control_validation_suite_phase1_exploration/full_control_seed42/trials.ndjson"
    )
    reference_log: Path = Path("data/ibl/reference.ndjson")
    output: Path = Path("docs/figures/agent_vs_animal_full_control_seed42.png")


def main(args: Args) -> None:
    apply_journal_style()

    df_agent = load_trials(args.agent_log)
    df_animal = load_trials(args.reference_log)
    if df_agent.empty or df_animal.empty:
        raise SystemExit("Empty trial log on agent or reference side.")

    task_name = df_agent["task"].iloc[0]
    m_agent = compute_all_metrics(df_agent, task_name)
    m_animal = compute_all_metrics(df_animal, task_name)

    stim_col, _ = _stim_column(df_agent)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    plt.subplots_adjust(wspace=0.34, left=0.06, right=0.985, top=0.9, bottom=0.16)

    _plot_psychometric(axes[0], df_agent, df_animal, m_agent, m_animal, stim_col)
    _plot_chronometric(axes[1], df_agent, df_animal, stim_col)
    _plot_history(axes[2], m_agent, m_animal)

    for ax, label in zip(axes, "abc"):
        add_panel_label(ax, label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
