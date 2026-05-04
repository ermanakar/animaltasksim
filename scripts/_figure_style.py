"""Shared matplotlib style for AnimalTaskSim README figures.

Approximates the conventions of high-impact-journal figure layouts:
sans-serif typography, no top/right spines, outward ticks, no panel
titles inside the axes (descriptive captions live in the README), and
bold panel labels (a, b, c, ...) in the top-left corner.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Two-tone agent-vs-reference palette (Fig 1).
COLOR_AGENT = "#1f6f9f"
COLOR_REFERENCE = "#3d3d3d"

# Muted condition palette for lesion comparisons (Figs 2 and 3).
CONDITION_COLORS = {
    "true_no_control": "#6c7480",
    "exploration_only": "#a04781",
    "persistence_only": "#2f8a55",
    "full_control": "#1f6f9f",
}

DELTA_COLOR_RETRY = "#2f8a55"
DELTA_COLOR_STALE = "#a04781"


def apply_journal_style() -> None:
    """Apply a publication-style matplotlib rcParams update."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "normal",
            "axes.titlepad": 6.0,
            "axes.labelsize": 10.0,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.borderpad": 0.2,
            "savefig.dpi": 220,
            "savefig.facecolor": "white",
            "figure.dpi": 130,
            "figure.facecolor": "white",
        }
    )


def add_panel_label(ax: Axes, label: str, *, x: float = -0.14, y: float = 1.06) -> None:
    """Add a bold panel label (a, b, c, ...) in the top-left of an axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111111",
    )
