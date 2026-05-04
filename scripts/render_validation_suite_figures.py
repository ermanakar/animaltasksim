#!/usr/bin/env python
"""Render the README's Figures 2 and 3 from validation_summary.json.

Figure 2: 4-panel suite summary (retry gap, stale-switch lift,
psychometric slope, chronometric slope) across the four lesion conditions.

Figure 3: paired-delta bar chart for the three adaptive conditions vs.
the no-control lesion, with positive-seed counts annotated above/below
each bar.

Both figures share the journal-style helpers in `_figure_style.py`:
sans-serif typography, no top/right spines, no in-axes panel titles,
and bold panel labels in the top-left.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tyro

from _figure_style import (  # noqa: E402
    CONDITION_COLORS,
    DELTA_COLOR_RETRY,
    DELTA_COLOR_STALE,
    add_panel_label,
    apply_journal_style,
)

CONDITION_ORDER = [
    "true_no_control",
    "exploration_only",
    "persistence_only",
    "full_control",
]
CONDITION_LABELS = {
    "true_no_control": "No control",
    "exploration_only": "Exploration\nonly",
    "persistence_only": "Persistence\nonly",
    "full_control": "Full\ncontrol",
}

DELTA_COMPARISONS = [
    ("exploration_only_minus_true_no_control", "Exploration only"),
    ("persistence_only_minus_true_no_control", "Persistence only"),
    ("full_control_minus_true_no_control", "Full control"),
]

IBL_PSYCH_REF_MEAN = 20.0
IBL_PSYCH_REF_STD = 5.7
IBL_CHRONO_LITERATURE = -36.0

ERRORBAR_KW = dict(ecolor="#333333", elinewidth=0.9, capsize=3.0, capthick=0.9)


def _bar_panel(
    ax,
    aggregate: dict[str, dict],
    mean_key: str,
    std_key: str,
    ylabel: str,
) -> None:
    means = [aggregate[c][mean_key] for c in CONDITION_ORDER]
    stds = [aggregate[c][std_key] for c in CONDITION_ORDER]
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]

    xs = list(range(len(CONDITION_ORDER)))
    ax.bar(
        xs, means,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        width=0.66,
        error_kw=ERRORBAR_KW,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="#888888", linewidth=0.6, zorder=0)


def render_summary(summary: dict, output: Path) -> Path:
    aggregate = {row["condition"]: row for row in summary["aggregate"]}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))
    plt.subplots_adjust(
        wspace=0.30, hspace=0.55, left=0.07, right=0.985, top=0.95, bottom=0.10
    )

    _bar_panel(
        axes[0, 0],
        aggregate,
        "retry_gap_mean",
        "retry_gap_std",
        "Retry gap\nP(retry|weak fail) − P(retry|strong fail)",
    )

    _bar_panel(
        axes[0, 1],
        aggregate,
        "stale_switch_lift_overall_mean",
        "stale_switch_lift_overall_std",
        "Stale-switch lift\nP(switch|stale) − P(switch|fresh)",
    )

    psych_ax = axes[1, 0]
    _bar_panel(
        psych_ax,
        aggregate,
        "psychometric_slope_mean",
        "psychometric_slope_std",
        "Psychometric slope\n(logits / contrast)",
    )
    psych_ax.axhspan(
        IBL_PSYCH_REF_MEAN - IBL_PSYCH_REF_STD,
        IBL_PSYCH_REF_MEAN + IBL_PSYCH_REF_STD,
        color="#1565c0", alpha=0.10, zorder=0,
        label=f"IBL ref {IBL_PSYCH_REF_MEAN:.0f} ± {IBL_PSYCH_REF_STD:.1f}",
    )
    psych_ax.axhline(IBL_PSYCH_REF_MEAN, color="#1565c0",
                     linewidth=0.9, linestyle="--", alpha=0.6)
    psych_ax.legend(loc="upper right")

    chrono_ax = axes[1, 1]
    _bar_panel(
        chrono_ax,
        aggregate,
        "chronometric_slope_mean",
        "chronometric_slope_std",
        "Chronometric slope\n(ms per unit |stimulus|)",
    )
    chrono_ax.axhline(
        IBL_CHRONO_LITERATURE,
        color="#444444", linewidth=0.9, linestyle="--", alpha=0.7,
        label=f"Literature target ≈ {IBL_CHRONO_LITERATURE:.0f}",
    )
    chrono_ax.legend(loc="lower right")

    for ax, label in zip(axes.flat, "abcd"):
        add_panel_label(ax, label)

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_paired_deltas(summary: dict, output: Path) -> Path:
    paired = {row["comparison"]: row for row in summary["paired_delta_summary"]}

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    plt.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.18)

    width = 0.34
    xs = list(range(len(DELTA_COMPARISONS)))
    retry_means, retry_pos, stale_means, stale_pos, n_seeds, labels = (
        [], [], [], [], [], []
    )
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

    ax.bar(retry_xs, retry_means, width=width, color=DELTA_COLOR_RETRY,
           edgecolor="white", linewidth=0.8, label="Δ retry gap")
    ax.bar(stale_xs, stale_means, width=width, color=DELTA_COLOR_STALE,
           edgecolor="white", linewidth=0.8, label="Δ stale-switch lift")

    pad = max(abs(min(stale_means)), abs(max(retry_means))) * 0.22
    ymin, ymax = min(stale_means) - pad, max(retry_means) + pad
    ax.set_ylim(ymin, ymax)

    label_offset = (ymax - ymin) * 0.025

    def _annotate(x, value, count, n):
        offset = label_offset if value >= 0 else -label_offset
        va = "bottom" if value >= 0 else "top"
        ax.text(x, value + offset, f"{count}/{n}",
                ha="center", va=va, fontsize=8.5, color="#222222")

    for x, m, c, n in zip(retry_xs, retry_means, retry_pos, n_seeds):
        _annotate(x, m, c, n)
    for x, m, c, n in zip(stale_xs, stale_means, stale_pos, n_seeds):
        _annotate(x, m, c, n)

    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Paired Δ vs. no-control (mean across seeds)")
    ax.legend(loc="lower left")

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


@dataclass(slots=True)
class Args:
    run_root: Path = Path("runs/adaptive_control_validation_suite_phase1_exploration")
    output_dir: Path = Path("docs/figures")


def main(args: Args) -> None:
    apply_journal_style()
    summary_path = args.run_root / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_png = render_summary(summary, args.output_dir / "suite_validation_summary.png")
    deltas_png = render_paired_deltas(summary, args.output_dir / "suite_paired_deltas.png")
    print(f"Wrote {summary_png}")
    print(f"Wrote {deltas_png}")


if __name__ == "__main__":
    main(tyro.cli(Args))
