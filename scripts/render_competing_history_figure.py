"""Render the competing-history comparison from its portable result snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import tyro


@dataclass(slots=True)
class Args:
    """Portable result and exported figure location."""

    source: Path = Path("docs/results/competing_history_v1.json")
    output: Path = Path("docs/figures/competing_history_v1.png")


def main(args: Args) -> None:
    """Render fixed means and descriptive intervals without any model fitting."""
    payload = json.loads(args.source.read_text())
    a = payload["subject_folds_results"]
    b = payload["lab_folds_results"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    models = [
        "outcome_history",
        "long_history",
        "combined_history",
        "combined_evidence",
    ]
    labels = [
        "One-step history",
        "Five-trial history",
        "History + learned side",
        "Combined + evidence",
    ]
    axes[0].barh(
        labels,
        [a["equal_animal_nll"][m] for m in models],
        color=["#94a3b8", "#64748b", "#475569", "#087f6d"],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.43)
    axes[0].set_xlabel("Choice NLL (lower is better)")
    axes[0].set_title("Prediction with stronger controls")
    keys = [
        "evidence_history_over_outcome_history",
        "long_evidence_over_long_history",
        "learning_evidence_over_learning_history",
        "combined_evidence_over_combined_history",
    ]
    for j, (result, color, label) in enumerate(
        [(a, "#087f6d", "Animal folds"), (b, "#7152a3", "Leave one lab out")]
    ):
        for i, k in enumerate(keys):
            c = result["contrasts"][k]
            m = c["mean_gain"]
            lo, hi = c["subject_bootstrap_95"]
            axes[1].errorbar(
                m,
                i + (j - 0.5) * 0.14,
                xerr=[[m - lo], [hi - m]],
                fmt="o",
                color=color,
                capsize=3,
                label=label if i == 0 else None,
            )
    axes[1].set_yticks(
        range(4),
        [
            "One-step controls",
            "Five-trial controls",
            "Learning controls",
            "Combined controls",
        ],
    )
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#777", lw=0.8)
    axes[1].set_xlabel("Added evidence-history gain (nats/trial)")
    axes[1].set_title("Smaller residual gain after longer history")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Development analysis: 83 mice, 9 labs, identical trial samples", fontsize=13
    )
    fig.text(
        0.02,
        0.015,
        "Intervals: descriptive animal bootstrap, conditional on fitted folds; not lab-level confidence intervals.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    for ext in ["png", "svg"]:
        p = args.output.with_suffix(f".{ext}")
        fig.savefig(p, dpi=170)
        if ext == "svg":
            p.write_text(
                "\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(Args))
