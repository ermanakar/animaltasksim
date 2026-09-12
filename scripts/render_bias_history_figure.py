"""Render preference-control results and illustrative synthetic diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import tyro


@dataclass(slots=True)
class Args:
    """Portable snapshot and output figure."""

    source: Path = Path("docs/results/bias_history_v1.json")
    output: Path = Path("docs/figures/bias_history_v1.png")


def main(args: Args) -> None:
    """Render saved scores without fitting or accessing animal records."""
    payload = json.loads(args.source.read_text())
    real = payload["results"]
    summary = payload["recovery_summary"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    models = ["one_step", "bias", "long", "bias_long", "bias_long_evidence"]
    labels = [
        "One-step history",
        "Preference controls",
        "Five-trial history",
        "Preference + history",
        "Preference + history + evidence",
    ]
    axes[0].barh(
        labels,
        [real["equal_animal_nll"][m] for m in models],
        color=["#94a3b8", "#64748b", "#64748b", "#475569", "#087f6d"],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.42)
    axes[0].set_xlabel("Equal-animal choice NLL (lower is better)")
    axes[0].set_title("Real data: 83 development mice")
    counts = np.array(
        [
            [
                row[key]["positive_lower_intervals"]
                for key in ["bias_long_over_bias", "bias_long_evidence_over_bias_long"]
            ]
            for row in summary
        ]
    )
    axes[1].imshow(counts, vmin=0, vmax=5, cmap="Blues", aspect="auto")
    axes[1].set_yticks(
        range(4),
        [
            "Static bias only",
            "Slow drift only",
            "True history",
            "Bias + drift + history",
        ],
    )
    axes[1].set_xticks([0, 1], ["Added lags", "Added evidence"])
    for i in range(4):
        for j in range(2):
            axes[1].text(
                j,
                i,
                f"{counts[i, j]}/5",
                ha="center",
                va="center",
                color="white" if counts[i, j] >= 3 else "black",
            )
    axes[1].set_title("Synthetic runs with positive lower interval")
    axes[0].spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Prediction improves; causal explanations remain ambiguous", fontsize=13
    )
    fig.text(
        0.02,
        0.015,
        "Five synthetic seeds per regime are illustrative, not a false-positive or power estimate. Drift-only lag gains do not imply true history updating.",
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
