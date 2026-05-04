#!/usr/bin/env python
"""Render a standalone agent-vs-animal comparison PNG for the README.

Reuses the comparison plot helpers from `eval.dashboard` so the figure shows
the same psychometric / chronometric / history panels as the per-run dashboard,
but as one polished PNG instead of an embedded HTML asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tyro

from eval.dashboard import (
    _plot_chronometric_comparison,
    _plot_history_comparison,
    _plot_psychometric_comparison,
)
from eval.metrics import compute_all_metrics, load_trials


@dataclass(slots=True)
class Args:
    agent_log: Path = Path(
        "runs/adaptive_control_validation_suite_phase1_exploration/full_control_seed42/trials.ndjson"
    )
    reference_log: Path = Path("data/ibl/reference.ndjson")
    output: Path = Path("docs/figures/agent_vs_animal_full_control_seed42.png")


def main(args: Args) -> None:
    df_agent = load_trials(args.agent_log)
    df_animal = load_trials(args.reference_log)
    if df_agent.empty or df_animal.empty:
        raise SystemExit("Empty trial log on agent or reference side.")

    task_name = df_agent["task"].iloc[0]
    metrics_agent = compute_all_metrics(df_agent, task_name)
    metrics_animal = compute_all_metrics(df_animal, task_name)

    if "stimulus_contrast" in df_agent.columns:
        stim_column = "stimulus_contrast"
    elif "stimulus_coherence" in df_agent.columns:
        stim_column = "stimulus_coherence"
    else:
        raise SystemExit("No stimulus column found in agent data.")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    fig.suptitle(
        "Adaptive-control agent vs. IBL mouse — full control, seed 42",
        fontsize=13,
        fontweight="bold",
        color="#2b2b2b",
    )

    _plot_psychometric_comparison(
        axes[0], df_agent, df_animal, metrics_agent, metrics_animal, stim_column
    )
    _plot_chronometric_comparison(
        axes[1], df_agent, df_animal, metrics_agent, metrics_animal
    )
    _plot_history_comparison(axes[2], metrics_agent, metrics_animal)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
