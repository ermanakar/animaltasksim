#!/usr/bin/env python3
"""
Query and manage the experiment registry.

Usage examples:
    # List all experiments
    python scripts/query_registry.py list
    
    # Filter by task
    python scripts/query_registry.py list --task rdm_macaque
    
    # Filter by agent
    python scripts/query_registry.py list --agent hybrid_ddm_lstm
    
    # Show details for a specific run
    python scripts/query_registry.py show ibl_stickyq_latency
    
    # Export to CSV
    python scripts/query_registry.py export registry.csv
"""

from pathlib import Path
from typing import Annotated, Literal

import tyro
from rich.console import Console
from rich.table import Table

from animaltasksim.registry import ExperimentRegistry

console = Console()


def list_experiments(
    registry_path: Path = Path("runs/registry.json"),
    task: str | None = None,
    agent: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """
    List experiments in the registry.
    
    Args:
        registry_path: Path to registry JSON file
        task: Filter by task type
        agent: Filter by agent type
        status: Filter by status
        tags: Filter by tags
    """
    registry = ExperimentRegistry(registry_path)
    experiments = registry.filter(task=task, agent=agent, status=status, tags=tags)
    
    if not experiments:
        console.print("[yellow]No experiments found matching criteria.[/yellow]")
        return
    
    # Create rich table
    table = Table(title=f"Experiment Registry ({len(experiments)} results)")
    
    table.add_column("Run ID", style="cyan", no_wrap=False)
    table.add_column("Date", style="magenta")
    table.add_column("Task", style="green")
    table.add_column("Agent", style="blue")
    table.add_column("Status", style="yellow")
    table.add_column("Metrics", style="white")
    
    for exp in sorted(experiments, key=lambda e: e.created_date, reverse=True):
        # Format metrics summary
        metrics_parts = []
        if exp.psychometric_slope is not None:
            metrics_parts.append(f"Psych: {exp.psychometric_slope:.2f}")
        if exp.chronometric_slope is not None:
            metrics_parts.append(f"Chron: {exp.chronometric_slope:.0f}")
        if exp.win_stay_rate is not None:
            metrics_parts.append(f"WS: {exp.win_stay_rate:.2f}")
        
        metrics_str = "\n".join(metrics_parts) if metrics_parts else "—"
        
        table.add_row(
            exp.run_id,
            exp.created_date,
            exp.task,
            exp.agent,
            exp.status,
            metrics_str,
        )
    
    console.print(table)


def show_experiment(
    run_id: str,
    registry_path: Path = Path("runs/registry.json"),
) -> None:
    """
    Show detailed information for a specific experiment.
    
    Args:
        run_id: Run ID to show
        registry_path: Path to registry JSON file
    """
    registry = ExperimentRegistry(registry_path)
    exp = registry.get(run_id)
    
    if exp is None:
        console.print(f"[red]Experiment '{run_id}' not found in registry.[/red]")
        return
    
    console.print(f"\n[bold cyan]Experiment: {exp.run_id}[/bold cyan]\n")
    
    # Basic info
    console.print(f"[bold]Basic Information[/bold]")
    console.print(f"  Date: {exp.created_date}")
    console.print(f"  Task: {exp.task}")
    console.print(f"  Agent: {exp.agent}")
    console.print(f"  Status: {exp.status}")
    if exp.variant:
        console.print(f"  Variant: {exp.variant}")
    if exp.seed is not None:
        console.print(f"  Seed: {exp.seed}")
    if exp.total_trials:
        console.print(f"  Total Trials: {exp.total_trials}")
    if exp.training_time_min:
        console.print(f"  Training Time: {exp.training_time_min:.1f} min")
    
    # Metrics
    console.print(f"\n[bold]Behavioral Metrics[/bold]")
    if exp.psychometric_slope is not None:
        console.print(f"  Psychometric Slope: {exp.psychometric_slope:.2f}")
    if exp.bias is not None:
        console.print(f"  Bias: {exp.bias:.4f}")
    if exp.chronometric_slope is not None:
        console.print(f"  Chronometric Slope: {exp.chronometric_slope:.0f} ms/unit")
    if exp.rt_intercept is not None:
        console.print(f"  RT Intercept: {exp.rt_intercept:.0f} ms")
    if exp.win_stay_rate is not None:
        console.print(f"  Win-Stay Rate: {exp.win_stay_rate:.3f}")
    if exp.lose_shift_rate is not None:
        console.print(f"  Lose-Shift Rate: {exp.lose_shift_rate:.3f}")
    if exp.sticky_choice_rate is not None:
        console.print(f"  Sticky Choice Rate: {exp.sticky_choice_rate:.3f}")
    
    # Files
    console.print(f"\n[bold]Files[/bold]")
    if exp.config_path:
        console.print(f"  Config: {exp.config_path}")
    if exp.log_path:
        console.print(f"  Logs: {exp.log_path}")
    if exp.metrics_path:
        console.print(f"  Metrics: {exp.metrics_path}")
    if exp.dashboard_path:
        console.print(f"  Dashboard: {exp.dashboard_path}")
    
    # Tags and notes
    if exp.tags:
        console.print(f"\n[bold]Tags[/bold]")
        console.print(f"  {', '.join(exp.tags)}")
    
    if exp.notes:
        console.print(f"\n[bold]Notes[/bold]")
        console.print(f"  {exp.notes}")
    
    console.print()


def export_csv(
    output_path: Path,
    registry_path: Path = Path("runs/registry.json"),
) -> None:
    """
    Export registry to CSV.
    
    Args:
        output_path: Output CSV file path
        registry_path: Path to registry JSON file
    """
    registry = ExperimentRegistry(registry_path)
    registry.export_csv(output_path)
    console.print(f"[green]Exported {len(registry.experiments)} experiments to {output_path}[/green]")


def cli_list(
    registry_path: Path = Path("runs/registry.json"),
    task: str | None = None,
    agent: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """List experiments in the registry with optional filters."""
    list_experiments(registry_path, task, agent, status, tags)


def cli_show(
    run_id: str,
    registry_path: Path = Path("runs/registry.json"),
) -> None:
    """Show detailed information for a specific experiment."""
    show_experiment(run_id, registry_path)


def cli_export(
    output_path: Path,
    registry_path: Path = Path("runs/registry.json"),
) -> None:
    """Export registry to CSV."""
    export_csv(output_path, registry_path)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "list": cli_list,
        "show": cli_show,
        "export": cli_export,
    })
