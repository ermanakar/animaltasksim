"""CLI tool to generate comparison dashboards for AnimalTaskSim runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from eval.dashboard import build_comparison_dashboard


@dataclass
class DashboardOptions:
    """Options for generating comparison dashboards."""

    agent_log: str
    """Path to agent's NDJSON log file (e.g., runs/ibl_stickyq_v18/train.ndjson)."""

    reference_log: str
    """Path to animal reference NDJSON file (e.g., data/ibl/reference.ndjson)."""

    output: str
    """Path to save the HTML dashboard (e.g., runs/ibl_stickyq_v18/dashboard.html)."""

    agent_name: str = "Agent"
    """Name to display for the agent (e.g., 'Sticky-GLM v18')."""

    reference_name: str = "Animal Reference"
    """Name to display for the reference data (e.g., 'IBL Reference')."""

    title: str = "Agent vs Animal Behavioral Comparison"
    """Dashboard title."""


def main(opts: DashboardOptions) -> None:
    """
    Generate an interactive HTML dashboard comparing agent and animal behavior.

    This creates a side-by-side comparison with:
    - Psychometric curves (stimulus-response relationship)
    - Chronometric curves (reaction time vs difficulty)
    - History effects (win-stay, lose-shift, sticky-choice)
    - Accuracy by coherence/contrast
    - Quantitative metrics table with match percentages

    Examples
    --------
    Compare mouse Sticky-GLM to IBL reference:
        python scripts/make_dashboard.py \\
            --agent-log runs/ibl_stickyq_v18/train.ndjson \\
            --reference-log data/ibl/reference.ndjson \\
            --output runs/ibl_stickyq_v18/dashboard.html \\
            --agent-name "Sticky-GLM v18" \\
            --reference-name "IBL Reference"

    Compare macaque PPO to Roitman reference:
        python scripts/make_dashboard.py \\
            --agent-log runs/rdm_ppo_v18_confidence/train.ndjson \\
            --reference-log data/macaque/reference.ndjson \\
            --output runs/rdm_ppo_v18_confidence/dashboard.html \\
            --agent-name "PPO v18" \\
            --reference-name "Roitman & Shadlen 2002"
    """
    agent_path = Path(opts.agent_log)
    reference_path = Path(opts.reference_log)
    output_path = Path(opts.output)

    if not agent_path.exists():
        raise FileNotFoundError(f"Agent log not found: {agent_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference log not found: {reference_path}")

    print(f"Building dashboard...")
    print(f"  Agent: {agent_path}")
    print(f"  Reference: {reference_path}")
    print(f"  Output: {output_path}")

    build_comparison_dashboard(
        agent_log_path=agent_path,
        animal_log_path=reference_path,
        output_path=output_path,
        title=opts.title,
        agent_name=opts.agent_name,
        animal_name=opts.reference_name,
    )

    print(f"\n✓ Dashboard generated successfully!")
    print(f"  Open in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    tyro.cli(main)
