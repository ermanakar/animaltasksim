#!/usr/bin/env python
"""Render polished suite figures from a validation_summary.json file.

Produces:
  - <run_root>/suite_validation_summary.png
  - <run_root>/suite_paired_deltas.png
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tyro

CONDITION_ORDER = [
    "true_no_control",
    "exploration_only",
    "persistence_only",
    "full_control",
]
CONDITION_LABELS = {
    "true_no_control": "No control",
    "exploration_only": "Exploration only",
    "persistence_only": "Persistence only",
    "full_control": "Full control",
}
CONDITION_COLORS = {
    "true_no_control": "#6c757d",
    "exploration_only": "#c2185b",
    "persistence_only": "#2e7d32",
    "full_control": "#1565c0",
}

DELTA_COMPARISONS = [
    ("exploration_only_minus_true_no_control", "Exploration only"),
    ("persistence_only_minus_true_no_control", "Persistence only"),
    ("full_control_minus_true_no_control", "Full control"),
]

IBL_PSYCH_REF_MEAN = 20.0
IBL_PSYCH_REF_STD = 5.7
IBL_CHRONO_LITERATURE = -36.0

TEXT_COLOR = "#2b2b2b"
GRID_KW = dict(color="#d9d9d9", linewidth=0.7, alpha=0.9)
ERRORBAR_KW = dict(ecolor="#333333", elinewidth=1.0, capsize=4, capthick=1.0)


def _apply_axes_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.yaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)


def _bar_panel(
    ax: plt.Axes,
    aggregate: dict[str, dict],
    mean_key: str,
    std_key: str,
    title: str,
    ylabel: str,
    *,
    invert_y: bool = False,
) -> None:
    means = [aggregate[c][mean_key] for c in CONDITION_ORDER]
    stds = [aggregate[c][std_key] for c in CONDITION_ORDER]
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]

    xs = list(range(len(CONDITION_ORDER)))
    bars = ax.bar(
        xs,
        means,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        width=0.68,
        error_kw=ERRORBAR_KW,
    )
    _ = bars  # silence unused
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right", color=TEXT_COLOR)
    ax.set_title(title, fontsize=11, color=TEXT_COLOR, pad=10)
    ax.set_ylabel(ylabel, fontsize=9.5, color=TEXT_COLOR)
    ax.axhline(0, color="#888888", linewidth=0.8)
    if invert_y:
        ax.invert_yaxis()
    _apply_axes_style(ax)


def render_summary(summary: dict, output: Path) -> Path:
    aggregate = {row["condition"]: row for row in summary["aggregate"]}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
    fig.suptitle(
        "Adaptive Control Validation Suite — Phase 1",
        fontsize=14,
        color=TEXT_COLOR,
        weight="bold",
    )

    _bar_panel(
        axes[0, 0],
        aggregate,
        "retry_gap_mean",
        "retry_gap_std",
        "Persistence readout: retry gap",
        "P(retry | weak fail) − P(retry | strong fail)",
    )
    _bar_panel(
        axes[0, 1],
        aggregate,
        "stale_switch_lift_overall_mean",
        "stale_switch_lift_overall_std",
        "Exploration readout: stale-switch lift",
        "P(switch | stale) − P(switch | fresh)",
    )

    psych_ax = axes[1, 0]
    _bar_panel(
        psych_ax,
        aggregate,
        "psychometric_slope_mean",
        "psychometric_slope_std",
        "Core fingerprint: psychometric slope",
        "Slope (logits / contrast)",
    )
    psych_ax.axhspan(
        IBL_PSYCH_REF_MEAN - IBL_PSYCH_REF_STD,
        IBL_PSYCH_REF_MEAN + IBL_PSYCH_REF_STD,
        color="#1565c0",
        alpha=0.10,
        label=f"IBL ref {IBL_PSYCH_REF_MEAN:.0f} ± {IBL_PSYCH_REF_STD:.1f}",
    )
    psych_ax.axhline(IBL_PSYCH_REF_MEAN, color="#1565c0", linewidth=0.9, linestyle="--", alpha=0.7)
    psych_ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    chrono_ax = axes[1, 1]
    _bar_panel(
        chrono_ax,
        aggregate,
        "chronometric_slope_mean",
        "chronometric_slope_std",
        "Core fingerprint: chronometric slope",
        "ms per unit |stimulus|",
    )
    chrono_ax.axhline(
        IBL_CHRONO_LITERATURE,
        color="#444444",
        linewidth=0.9,
        linestyle="--",
        alpha=0.8,
        label=f"Literature target ≈ {IBL_CHRONO_LITERATURE:.0f}",
    )
    chrono_ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def render_paired_deltas(summary: dict, output: Path) -> Path:
    paired = {row["comparison"]: row for row in summary["paired_delta_summary"]}

    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    fig.suptitle(
        "Paired lesion deltas vs. no-control",
        fontsize=13,
        color=TEXT_COLOR,
        weight="bold",
    )

    width = 0.36
    xs = list(range(len(DELTA_COMPARISONS)))
    retry_means = []
    retry_pos = []
    stale_means = []
    stale_pos = []
    n_seeds = []
    labels = []
    for key, label in DELTA_COMPARISONS:
        row = paired[key]
        retry_means.append(row["delta_retry_gap_mean"])
        retry_pos.append(row["delta_retry_gap_positive_count"])
        stale_means.append(row["delta_stale_switch_lift_overall_mean"])
        stale_pos.append(row["delta_stale_switch_lift_overall_positive_count"])
        n_seeds.append(row["num_seeds"])
        labels.append(label)

    retry_xs = [x - width / 2 for x in xs]
    stale_xs = [x + width / 2 for x in xs]
    ax.bar(
        retry_xs,
        retry_means,
        width=width,
        color="#2e7d32",
        edgecolor="white",
        linewidth=1.0,
        label="Δ retry gap",
    )
    ax.bar(
        stale_xs,
        stale_means,
        width=width,
        color="#c2185b",
        edgecolor="white",
        linewidth=1.0,
        label="Δ stale-switch lift",
    )

    def _annotate(x: float, value: float, count: int, n: int) -> None:
        offset = 0.006 if value >= 0 else -0.006
        va = "bottom" if value >= 0 else "top"
        ax.text(
            x,
            value + offset,
            f"{count}/{n}",
            ha="center",
            va=va,
            fontsize=9,
            color=TEXT_COLOR,
        )

    for x, m, c, n in zip(retry_xs, retry_means, retry_pos, n_seeds):
        _annotate(x, m, c, n)
    for x, m, c, n in zip(stale_xs, stale_means, stale_pos, n_seeds):
        _annotate(x, m, c, n)

    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=TEXT_COLOR)
    ax.set_ylabel("Paired delta vs. no-control (mean across seeds)", fontsize=10, color=TEXT_COLOR)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)

    pad = max(abs(min(stale_means)), abs(max(retry_means))) * 0.18
    ymin = min(stale_means) - pad
    ymax = max(retry_means) + pad
    ax.set_ylim(ymin, ymax)

    _apply_axes_style(ax)
    ax.text(
        0.0,
        -0.16,
        "Labels show positive-seed counts (n/N). Effect is positive only when most seeds agree in sign.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


@dataclass(slots=True)
class Args:
    run_root: Path = Path("runs/adaptive_control_validation_suite_phase1_exploration")
    output_dir: Path = Path("docs/figures")


def main(args: Args) -> None:
    summary_path = args.run_root / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_png = render_summary(summary, args.output_dir / "suite_validation_summary.png")
    deltas_png = render_paired_deltas(summary, args.output_dir / "suite_paired_deltas.png")
    print(f"Wrote {summary_png}")
    print(f"Wrote {deltas_png}")


if __name__ == "__main__":
    main(tyro.cli(Args))
