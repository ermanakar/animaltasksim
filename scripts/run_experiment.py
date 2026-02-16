#!/usr/bin/env python
"""
Interactive Experiment Runner for AnimalTaskSim

Streamlined workflow: Train → Evaluate → Dashboard → Registry
All the bells and whistles in one command!
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
except ImportError:
    print("Missing dependencies. Please install with: pip install -e '.[dev]'")
    sys.exit(1)

console = Console()


# Agent descriptions
AGENT_INFO = {
    "ppo": {
        "name": "PPO (Proximal Policy Optimization)",
        "description": "Reinforcement learning baseline. Fast training, learns task structure well but may lack realistic RT dynamics.",
        "tasks": ["ibl_2afc", "rdm_macaque"],
        "training_time": "~2-5 minutes",
    },
    "sticky_q": {
        "name": "Sticky Q-Learning",
        "description": "GLM-based agent with history biases. Good for IBL task, captures win-stay/lose-shift behavior.",
        "tasks": ["ibl_2afc"],
        "training_time": "~1-2 minutes",
    },
    "bayes": {
        "name": "Bayesian Observer",
        "description": "Ideal observer with sensory noise and lapses. Provides theoretical baseline for psychometric curves.",
        "tasks": ["ibl_2afc", "rdm_macaque"],
        "training_time": "~1-2 minutes",
    },
    "hybrid_ddm_lstm": {
        "name": "Hybrid DDM+LSTM",
        "description": "State-of-the-art: Drift-diffusion model + recurrent policy. Realistic RT dynamics and behavioral fingerprints. Requires curriculum learning.",
        "tasks": ["rdm_macaque"],
        "training_time": "~5-15 minutes",
    },
    "r_ddm": {
        "name": "R-DDM (Recurrent Drift-Diffusion)",
        "description": "History-aware diffusion model trained on multi-session animal data. Optimises choice, RT, and win-stay/lose-shift statistics.",
        "tasks": ["ibl_2afc"],
        "training_time": "~5-10 minutes (data-driven)",
    },
}

TASK_INFO = {
    "ibl_2afc": {
        "name": "Mouse 2AFC (IBL)",
        "description": "International Brain Laboratory visual contrast discrimination. Mice judge left/right gratings with varying contrast.",
        "reference": "IBL (2021) Neuron",
        "reference_data": "data/ibl/reference.ndjson",
    },
    "rdm_macaque": {
        "name": "Macaque Random Dot Motion",
        "description": "Classic Shadlen lab task. Macaques judge motion direction of coherently moving dots.",
        "reference": "Britten et al. (1992); Palmer, Huk & Shadlen (2005)",
        "reference_data": "data/macaque/reference.ndjson",
    },
}


REFERENCE_TARGETS: dict[str, dict[str, str]] = {
    "ibl_2afc": {
        "psych_slope": "13.2 (IBL mouse)",
        "bias": "~0.07 (slight rightward)",
        "chrono_slope": "-36 ms/unit (mouse)",
        "rt_intercept": "300 ms (mouse)",
        "win_stay": "0.73 (IBL mouse)",
        "lose_shift": "0.34 (IBL mouse)",
        "sticky_choice": "0.69 (IBL mouse)",
    },
    "rdm_macaque": {
        "psych_slope": "17.6 (macaque)",
        "bias": "~0.0 (unbiased)",
        "chrono_slope": "-645 ms/unit (macaque)",
        "rt_intercept": "760 ms (macaque)",
        "win_stay": "0.46 (macaque)",
        "lose_shift": "0.52 (macaque)",
        "sticky_choice": "0.46 (macaque)",
    },
}


def _fmt(value: object, fmt: str = ".2f") -> str:
    """Safely format a metric value, returning 'N/A' for None/NaN."""
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN check
        return "N/A"
    return f"{f:{fmt}}"


def show_welcome():
    """Display welcome message."""
    console.print(Panel.fit(
        "[bold cyan]AnimalTaskSim - Interactive Experiment Runner[/bold cyan]\n\n"
        "This wizard will guide you through:\n"
        "  1. Select task and agent\n"
        "  2. Configure training parameters\n"
        "  3. Train agent\n"
        "  4. Evaluate behavioral metrics\n"
        "  5. Generate comparison dashboard\n"
        "  6. Update experiment registry\n"
        "  7. Review results\n",
        border_style="cyan"
    ))
    console.print()


def select_task() -> str:
    """Let user select a task."""
    console.print("[bold]Step 1: Select Task Environment[/bold]\n", style="green")
    
    # Show task options
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan", width=12)
    table.add_column("Task", width=30)
    table.add_column("Description", width=60)
    
    for i, (task_id, info) in enumerate(TASK_INFO.items(), 1):
        table.add_row(
            f"[{i}]",
            info["name"],
            f"{info['description']}\n[dim]Ref: {info['reference']}[/dim]"
        )
    
    console.print(table)
    console.print()
    
    choice = Prompt.ask(
        "Select task",
        choices=["1", "2"],
        default="2"
    )
    
    task = list(TASK_INFO.keys())[int(choice) - 1]
    console.print(f"✓ Selected: [bold cyan]{TASK_INFO[task]['name']}[/bold cyan]\n")
    return task


def select_agent(task: str) -> str:
    """Let user select an agent compatible with the task."""
    console.print("[bold]Step 2: Select Agent[/bold]\n", style="green")
    
    # Filter agents compatible with this task
    compatible_agents = {
        agent_id: info
        for agent_id, info in AGENT_INFO.items()
        if task in info["tasks"]
    }
    
    # Show agent options
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan", width=12)
    table.add_column("Agent", width=30)
    table.add_column("Description", width=50)
    table.add_column("Training Time", width=15)
    
    for i, (agent_id, info) in enumerate(compatible_agents.items(), 1):
        table.add_row(
            f"[{i}]",
            info["name"],
            info["description"],
            info["training_time"]
        )
    
    console.print(table)
    console.print()
    
    choices = [str(i) for i in range(1, len(compatible_agents) + 1)]
    choice = Prompt.ask(
        "Select agent",
        choices=choices,
        default="1"
    )
    
    agent = list(compatible_agents.keys())[int(choice) - 1]
    console.print(f"✓ Selected: [bold cyan]{AGENT_INFO[agent]['name']}[/bold cyan]\n")
    return agent


def configure_training(task: str, agent: str) -> dict:
    """Configure training parameters."""
    console.print("[bold]Step 3: Configure Training[/bold]\n", style="green")

    if agent == "r_ddm":
        default_epochs = 40
        default_rollout = 1200

        quick = Confirm.ask(
            "Use recommended defaults?",
            default=True
        )

        if quick:
            epochs = default_epochs
            max_sessions = None
            rollout_trials = default_rollout
            seed = 42
        else:
            epochs = int(Prompt.ask(
                "Number of training epochs",
                default=str(default_epochs)
            ))
            max_sessions_response = Prompt.ask(
                "Max sessions to sample (leave blank for all)",
                default=""
            ).strip()
            max_sessions = int(max_sessions_response) if max_sessions_response else None
            rollout_trials = int(Prompt.ask(
                "Number of rollout trials to log",
                default=str(default_rollout)
            ))
            seed = int(Prompt.ask(
                "Random seed (for reproducibility)",
                default="42"
            ))
    else:
        # Default values based on agent
        if agent == "hybrid_ddm_lstm":
            default_episodes = 10
            default_trials = 200
        elif agent == "ppo":
            default_episodes = 5
            default_trials = 200
        else:
            default_episodes = 3
            default_trials = 400

        # Quick or custom config
        quick = Confirm.ask(
            "Use recommended defaults?",
            default=True
        )

        if quick:
            episodes = default_episodes
            trials_per_episode = default_trials
            seed = 42
        else:
            episodes = int(Prompt.ask(
                "Number of episodes (training iterations)",
                default=str(default_episodes)
            ))
            trials_per_episode = int(Prompt.ask(
                "Trials per episode",
                default=str(default_trials)
            ))
            seed = int(Prompt.ask(
                "Random seed (for reproducibility)",
                default="42"
            ))
    
    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d")
    task_short = "ibl" if task == "ibl_2afc" else "rdm"
    agent_short = agent.replace("_", "")[:10]
    run_name = f"{timestamp}_{task_short}_{agent_short}"
    
    custom_name = Prompt.ask(
        "Run name (identifier for this experiment)",
        default=run_name
    )

    # Avoid overwriting existing runs
    output_dir = Path("runs") / custom_name
    if output_dir.exists():
        base = custom_name
        suffix = 2
        while (Path("runs") / f"{base}_v{suffix}").exists():
            suffix += 1
        new_name = f"{base}_v{suffix}"
        console.print(
            f"[yellow]Directory runs/{custom_name} already exists. "
            f"Using runs/{new_name} instead.[/yellow]"
        )
        custom_name = new_name

    if agent == "r_ddm":
        config = {
            "task": task,
            "agent": agent,
            "epochs": epochs,
            "max_sessions": max_sessions,
            "rollout_trials": rollout_trials,
            "seed": seed,
            "run_name": custom_name,
            "output_dir": f"runs/{custom_name}",
        }
    else:
        config = {
            "task": task,
            "agent": agent,
            "episodes": episodes,
            "trials_per_episode": trials_per_episode,
            "seed": seed,
            "run_name": custom_name,
            "output_dir": f"runs/{custom_name}",
        }
    
    # Summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Task: {TASK_INFO[task]['name']}")
    console.print(f"  Agent: {AGENT_INFO[agent]['name']}")
    if agent == "r_ddm":
        console.print(f"  Epochs: {epochs}")
        if max_sessions is not None:
            console.print(f"  Sessions sampled: {max_sessions}")
        else:
            console.print("  Sessions sampled: all available")
        console.print(f"  Rollout trials: {rollout_trials}")
        console.print(f"  Seed: {seed}")
    else:
        console.print(f"  Episodes: {episodes}")
        console.print(f"  Trials/episode: {trials_per_episode}")
        console.print(f"  Total trials: {episodes * trials_per_episode}")
        console.print(f"  Seed: {seed}")
    console.print(f"  Output: {config['output_dir']}\n")
    
    if not Confirm.ask("Proceed with training?", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        sys.exit(0)
    
    return config


def run_training(config: dict) -> bool:
    """Run the training process."""
    console.print("\n[bold]Step 3: Training Agent[/bold]\n", style="green")
    
    task = config["task"]
    agent = config["agent"]
    
    # Build command based on agent
    py = sys.executable
    if agent == "hybrid_ddm_lstm":
        # Use curriculum training for hybrid agent
        cmd = [
            py, "scripts/train_hybrid_curriculum.py",
            "--reference-log", TASK_INFO[task]["reference_data"],
            "--output-dir", config["output_dir"],
            "--seed", str(config["seed"]),
            "--episodes", str(config["episodes"]),
            "--trials-per-episode", str(config["trials_per_episode"]),
        ]
    elif agent == "r_ddm":
        cmd = [
            py, "scripts/train_r_ddm.py",
            "--run-dir", config["output_dir"],
            "--task", task,
            "--epochs", str(config["epochs"]),
            "--seed", str(config["seed"]),
            "--rollout-trials", str(config["rollout_trials"]),
        ]
        if config.get("max_sessions") is not None:
            cmd.extend(["--max-sessions", str(config["max_sessions"])])
    else:
        # Use standard training
        # Map task names to env names expected by train_agent.py
        env_map = {
            "ibl_2afc": "ibl_2afc",
            "rdm_macaque": "rdm",  # train_agent.py expects "rdm" not "rdm_macaque"
        }
        env_name = env_map[task]
        cmd = [
            py, "scripts/train_agent.py",
            "--env", env_name,
            "--agent", agent,
            "--episodes", str(config["episodes"]),
            "--trials-per-episode", str(config["trials_per_episode"]),
            "--seed", str(config["seed"]),
            "--out", config["output_dir"],
        ]

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")

    # Stream training output so long runs show progress
    proc = subprocess.run(cmd, capture_output=False, text=True, stderr=subprocess.PIPE)

    if proc.returncode == 0:
        console.print("\n✓ [bold green]Training completed successfully![/bold green]\n")
        best_log = Path(config["output_dir"]) / "trials_best.ndjson"
        if best_log.exists():
            config["best_log"] = str(best_log)
        return True
    else:
        console.print("✗ [bold red]Training failed![/bold red]")
        if proc.stderr:
            console.print(f"[dim]{proc.stderr}[/dim]\n")
        return False


def run_evaluation(config: dict) -> bool:
    """Run evaluation to generate metrics."""
    console.print("[bold]Step 4: Evaluating Agent[/bold]\n", style="green")
    
    use_best = config.get("best_log") is not None
    cmd = [sys.executable, "scripts/evaluate_agent.py", "--run", config["output_dir"]]
    if use_best:
        cmd.append("--use-best")

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")

    with console.status("[bold cyan]Computing behavioral metrics...", spinner="dots"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        console.print("✓ [bold green]Evaluation completed![/bold green]\n")
        return True
    else:
        console.print("✗ [bold red]Evaluation failed![/bold red]")
        output = (result.stdout or "") + (result.stderr or "")
        if output.strip():
            console.print(f"[dim]{output.strip()}[/dim]\n")
        return False


def generate_dashboard(config: dict) -> bool:
    """Generate interactive dashboard."""
    console.print("[bold]Step 5: Generating Dashboard[/bold]\n", style="green")
    
    task = config["task"]
    output_dir = Path(config["output_dir"])
    if config.get("best_log"):
        trials_log = config["best_log"]
    else:
        trials_log = str(output_dir / "trials.ndjson")
    reference_log = TASK_INFO[task]["reference_data"]
    dashboard_path = str(output_dir / "dashboard.html")

    cmd = [
        sys.executable, "scripts/make_dashboard.py",
        "--opts.agent-log", trials_log,
        "--opts.reference-log", reference_log,
        "--opts.output", dashboard_path,
    ]
    # Only pass reference metrics if the file actually exists
    ibl_ref_metrics = Path("out/ibl_reference_metrics.json")
    if task == "ibl_2afc" and ibl_ref_metrics.exists():
        cmd.extend(["--opts.reference-metrics", str(ibl_ref_metrics)])

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")

    with console.status("[bold cyan]Creating visualizations...", spinner="dots"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        console.print("✓ [bold green]Dashboard generated![/bold green]")
        console.print(f"  View at: [cyan]file://{Path(dashboard_path).absolute()}[/cyan]\n")
        return True
    else:
        console.print("✗ [bold red]Dashboard generation failed![/bold red]")
        output = (result.stdout or "") + (result.stderr or "")
        if output.strip():
            console.print(f"[dim]{output.strip()}[/dim]\n")
        return False


def update_registry() -> bool:
    """Update experiment registry."""
    console.print("[bold]Step 6: Updating Registry[/bold]\n", style="green")
    
    cmd = [sys.executable, "scripts/scan_runs.py", "--overwrite"]

    with console.status("[bold cyan]Scanning experiments...", spinner="dots"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        console.print("✓ [bold green]Registry updated![/bold green]\n")
        return True
    else:
        console.print("✗ [bold red]Registry update failed![/bold red]")
        output = (result.stdout or "") + (result.stderr or "")
        if output.strip():
            console.print(f"[dim]{output.strip()}[/dim]\n")
        return False


def show_results(config: dict):
    """Display final results and metrics."""
    console.print("[bold]Step 7: Results Summary[/bold]\n", style="green")
    
    # Load metrics
    metrics_file = Path(config["output_dir"]) / "metrics.json"
    if not metrics_file.exists():
        console.print("[yellow]No metrics file found.[/yellow]")
        return
    
    with open(metrics_file) as f:
        data = json.load(f)
        metrics = data.get("metrics", {})
    
    # Look up task-appropriate reference values
    task = config["task"]
    ref = REFERENCE_TARGETS.get(task, REFERENCE_TARGETS["rdm_macaque"])

    # Display metrics
    table = Table(title="Behavioral Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", width=20)
    table.add_column("Target (Reference)", width=30)

    # Psychometric
    if "psychometric" in metrics:
        psych = metrics["psychometric"]
        table.add_row(
            "Psychometric Slope",
            _fmt(psych.get("slope")),
            ref["psych_slope"],
        )
        table.add_row(
            "Bias",
            _fmt(psych.get("bias"), ".4f"),
            ref["bias"],
        )

    # Chronometric
    if "chronometric" in metrics:
        chron = metrics["chronometric"]
        slope_str = _fmt(chron.get("slope_ms_per_unit"), ".0f")
        table.add_row(
            "Chronometric Slope",
            f"{slope_str} ms/unit" if slope_str != "N/A" else "N/A",
            ref["chrono_slope"],
        )
        intercept_str = _fmt(chron.get("intercept_ms"), ".0f")
        table.add_row(
            "RT Intercept",
            f"{intercept_str} ms" if intercept_str != "N/A" else "N/A",
            ref["rt_intercept"],
        )

    # History
    if "history" in metrics:
        hist = metrics["history"]
        table.add_row(
            "Win-Stay Rate",
            _fmt(hist.get("win_stay"), ".3f"),
            ref["win_stay"],
        )
        table.add_row(
            "Lose-Shift Rate",
            _fmt(hist.get("lose_shift"), ".3f"),
            ref["lose_shift"],
        )
        table.add_row(
            "Sticky Choice",
            _fmt(hist.get("sticky_choice"), ".3f"),
            ref["sticky_choice"],
        )
    
    console.print(table)
    console.print()
    
    # Query command for later
    console.print("[bold]Access this run later:[/bold]")
    console.print(f"  python scripts/query_registry.py show --run-id {config['run_name']}\n")


def show_completion(config: dict):
    """Show completion message."""
    console.print(Panel.fit(
        f"[bold green]✨ Experiment Complete! ✨[/bold green]\n\n"
        f"Run ID: [cyan]{config['run_name']}[/cyan]\n"
        f"Location: [cyan]{config['output_dir']}[/cyan]\n\n"
        f"Generated files:\n"
        f"  • config.json - Training configuration\n"
        f"  • trials.ndjson - Trial-by-trial logs\n"
        f"  • metrics.json - Behavioral metrics\n"
        f"  • dashboard.html - Interactive visualization\n"
        f"  • model files - Trained weights\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"  • View dashboard in browser\n"
        f"  • Query registry: [cyan]python scripts/query_registry.py list[/cyan]\n"
        f"  • Compare runs: [cyan]python scripts/query_registry.py export --output results.csv[/cyan]\n\n"
        f"[dim]All data saved to registry for future reference.[/dim]",
        border_style="green"
    ))


def main():
    """Main workflow."""
    show_welcome()
    
    # Step 1: Select task
    task = select_task()
    
    # Step 2: Select agent
    agent = select_agent(task)
    
    # Step 3: Configure
    config = configure_training(task, agent)
    
    # Step 4: Train
    if not run_training(config):
        sys.exit(1)
    
    # Step 5: Evaluate
    if not run_evaluation(config):
        console.print("[yellow]Warning: Evaluation failed, continuing...[/yellow]\n")
    
    # Step 6: Dashboard
    if not generate_dashboard(config):
        console.print("[yellow]Warning: Dashboard generation failed, continuing...[/yellow]\n")
    
    # Step 7: Registry
    if not update_registry():
        console.print("[yellow]Warning: Registry update failed, continuing...[/yellow]\n")
    
    # Step 8: Results
    show_results(config)
    
    # Done!
    show_completion(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n\n[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)
