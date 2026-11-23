"""Render the November 2026 exploration and outcome-split figures from tracked snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

__all__ = ["Args", "main"]

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#777777"
CONTEXTS = {"reward_hard": "Reward,\nhard trial", "reward_easy": "Reward,\neasy trial",
            "error_hard": "Error,\nhard trial", "error_easy": "Error,\neasy trial"}
CONTROLS = {"outcome_history": "One-step history", "learning_history": "+ learned side",
            "long_history": "+ five-trial history", "combined_history": "+ both",
            "bias": "Side-bias controls", "bias_long": "Side-bias + five-trial"}


@dataclass(slots=True)
class Args:
    """Tracked inputs and exported figure directory; nothing is fitted here."""

    exploration: Path = Path("docs/results/deep_exploration_v1.json")
    split: Path = Path("docs/results/outcome_split_v1.json")
    output: Path = Path("docs/figures")


def _finish(fig: plt.Figure, axes: list[plt.Axes], path: Path, note: str) -> None:
    for ax in axes:
        ax.axhline(0, color=GREY, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.text(0.02, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main(args: Args) -> None:
    """Draw neighbour-trial weights with their acausal control, then the split evidence gains."""
    blocks = json.loads(args.exploration.read_text())["summary"]["history"]["blocks"]
    x = np.arange(len(CONTEXTS))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle("Most apparent choice history is shared with the following trial", fontsize=13, x=0.02, ha="left")
    for offset, key, color, label in [(-0.18, "past", BLUE, "Previous trial"), (0.18, "future", ORANGE, "Following trial")]:
        left.bar(x + offset, [blocks[key][c][0] for c in CONTEXTS], 0.34, color=color, label=label)
    net = np.array([blocks["net"][c][:3] for c in CONTEXTS])
    right.errorbar(x, net[:, 0], yerr=[net[:, 0] - net[:, 1], net[:, 2] - net[:, 0]], fmt="o", color=BLUE, capsize=4)
    for ax, title in [(left, "Weight on the neighbouring choice (log-odds)"), (right, "Previous minus following (95% interval)")]:
        ax.set_xticks(x, list(CONTEXTS.values()), fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
    left.legend(frameon=False, fontsize=8)
    _finish(fig, [left, right], args.output / "deep_exploration_v1.png",
            f"Block sessions, {blocks['animals']} mice, {blocks['trials']:,} trials; equal-animal means. "
            "Exploratory; the subtraction is an approximate control for slow side preference.")

    gains = json.loads(args.split.read_text())["all_sessions"]["gains"]
    x = np.arange(len(CONTROLS))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.suptitle("Weak-evidence history gain, split by previous outcome", fontsize=13, x=0.02, ha="left")
    for offset, key, color, label in [(-0.12, "reward_unique", BLUE, "After rewards"), (0.12, "error_unique", ORANGE, "After errors")]:
        rows = np.array([[gains[c][key]["mean_gain"], *gains[c][key]["subject_bootstrap_95"]] for c in CONTROLS]) * 1e3
        ax.errorbar(x + offset, rows[:, 0], yerr=[rows[:, 0] - rows[:, 1], rows[:, 2] - rows[:, 0]],
                    fmt="o", color=color, capsize=4, label=label)
    ax.set_xticks(x, list(CONTROLS.values()), fontsize=8)
    ax.set_ylabel("Unique NLL gain (10⁻³ nats/trial)")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, [ax], args.output / "outcome_split_v1.png",
            "83 development mice, animal-disjoint folds, subject-bootstrap 95% intervals. "
            "Exploratory; previously inspected animals.")


if __name__ == "__main__":
    main(tyro.cli(Args))
