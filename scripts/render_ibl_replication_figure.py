"""Render the frozen replication result without fitting or selecting models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass(slots=True)
class Args:
    """Saved result and exported scientific figure paths."""

    metrics: Path = Path("docs/results/replication_result_2026-09-11.json")
    output: Path = Path("docs/figures/ibl_replication_v1.png")


def main(args: Args) -> None:
    """Display the primary interval and every held-out animal's paired score."""
    payload = json.loads(args.metrics.read_text())
    result = payload.get("metrics", payload)
    delta = np.sort([row["nll"]["outcome_history"] - row["nll"]["evidence_history"]
                     for row in result["per_subject"]])
    mean = result["primary_equal_subject_nll_gain"]
    lo, hi = result["subject_bootstrap_95_interval"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [1, 3]}, sharex=True)
    fig.suptitle("Evidence-dependent history: reserved-cohort prediction", fontsize=13, x=0.12, ha="left")
    axes[0].errorbar(mean, 0, xerr=[[mean-lo], [hi-mean]], fmt="o", color="#25665f", capsize=5)
    axes[0].set_yticks([0], ["Equal-animal mean\n95% bootstrap interval"])
    axes[0].set_ylim(-0.7, 0.7)
    axes[0].set_title(f"{len(delta)} eligible mice · fixed models · no test-set fitting", fontsize=10, loc="left")
    axes[1].scatter(delta, np.arange(len(delta)), s=22, c=np.where(delta > 0, "#25665f", "#9d5845"))
    axes[1].set_yticks([])
    axes[1].set_ylabel("Each eligible mouse (ordered by gain)")
    axes[1].set_xlabel("Log-loss improvement over outcome history (nats/trial)\nPositive favors evidence-dependent history")
    for ax in axes:
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.18)
    fig.text(0.12, 0.015, "Internal prospective test; 15/60 candidates excluded by fixed QC.\nSubject bootstrap conditions on fixed fits; it does not establish generalization across labs.", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(Args))
