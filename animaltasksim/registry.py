"""
Experiment Registry System for AnimalTaskSim

Provides a JSON-based database for tracking runs, experiments, and their metadata.
Enables querying, filtering, and analyzing experiment history without navigating
scattered directory structures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExperimentMetadata(BaseModel):
    """Schema for experiment registry entries."""
    
    # Required fields
    run_id: str = Field(..., description="Unique identifier (usually directory name)")
    created_date: str = Field(..., description="ISO format date (YYYY-MM-DD)")
    task: Literal["ibl_2afc", "rdm_macaque"] = Field(..., description="Task environment")
    agent: Literal["sticky_q", "bayes_observer", "ppo", "ddm", "hybrid_ddm_lstm", "unknown"] = Field(..., description="Agent type")
    status: Literal["reference", "baseline", "experimental", "archived", "failed"] = Field(..., description="Run status")
    
    # Optional fields
    variant: str | None = Field(None, description="Experiment variant/description")
    seed: int | None = Field(None, description="Random seed used")
    total_trials: int | None = Field(None, description="Number of trials run")
    training_time_min: float | None = Field(None, description="Training duration in minutes")
    
    # Key metrics (from metrics.json)
    psychometric_slope: float | None = None
    bias: float | None = None
    chronometric_slope: float | None = None
    rt_intercept: float | None = None
    chronometric_slope_unit: str | None = None
    win_stay_rate: float | None = None
    lose_shift_rate: float | None = None
    sticky_choice_rate: float | None = None
    quality: dict[str, bool] | None = None
    
    # File paths (relative to project root)
    config_path: str | None = None
    log_path: str | None = None
    metrics_path: str | None = None
    dashboard_path: str | None = None
    
    # Notes
    notes: str | None = Field(None, description="Free-form notes about this run")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")


class ExperimentRegistry:
    """
    Manages the experiment registry database.
    
    The registry is stored as a JSON file with one entry per experiment run.
    Provides methods to add, update, query, and analyze experiments.
    """
    
    def __init__(self, registry_path: Path | str = "runs/registry.json"):
        """
        Initialize the registry.
        
        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.experiments: dict[str, ExperimentMetadata] = {}
        self.load()
    
    def load(self) -> None:
        """Load the registry from disk. Creates empty registry if file doesn't exist."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                self.experiments = {
                    run_id: ExperimentMetadata(**exp_data)
                    for run_id, exp_data in data.items()
                }
        else:
            self.experiments = {}
    
    def save(self) -> None:
        """Save the registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            data = {
                run_id: exp.model_dump(exclude_none=False)
                for run_id, exp in self.experiments.items()
            }
            json.dump(data, f, indent=2)
    
    def add(self, experiment: ExperimentMetadata, overwrite: bool = False) -> None:
        """
        Add or update an experiment in the registry.
        
        Args:
            experiment: Experiment metadata to add
            overwrite: If True, overwrites existing entry; if False, raises error on duplicate
        """
        if experiment.run_id in self.experiments and not overwrite:
            raise ValueError(f"Experiment '{experiment.run_id}' already exists. Use overwrite=True to update.")
        
        self.experiments[experiment.run_id] = experiment
        self.save()
    
    def get(self, run_id: str) -> ExperimentMetadata | None:
        """Get experiment by run_id."""
        return self.experiments.get(run_id)
    
    def delete(self, run_id: str) -> None:
        """Remove an experiment from the registry."""
        if run_id in self.experiments:
            del self.experiments[run_id]
            self.save()
    
    def list_all(self) -> list[ExperimentMetadata]:
        """Get all experiments."""
        return list(self.experiments.values())
    
    def filter(
        self,
        task: str | None = None,
        agent: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ExperimentMetadata]:
        """
        Filter experiments by criteria.
        
        Args:
            task: Filter by task type
            agent: Filter by agent type
            status: Filter by status
            tags: Filter by tags (must have ALL specified tags)
        
        Returns:
            List of matching experiments
        """
        results = self.list_all()
        
        if task:
            results = [e for e in results if e.task == task]
        if agent:
            results = [e for e in results if e.agent == agent]
        if status:
            results = [e for e in results if e.status == status]
        if tags:
            results = [e for e in results if all(tag in e.tags for tag in tags)]
        
        return results
    
    def summary_table(self) -> str:
        """Generate a human-readable summary table."""
        lines = [
            "Experiment Registry Summary",
            "=" * 80,
            f"{'Run ID':<40} {'Task':<12} {'Agent':<18} {'Status':<12}",
            "-" * 80,
        ]
        
        for exp in sorted(self.experiments.values(), key=lambda e: e.created_date, reverse=True):
            lines.append(
                f"{exp.run_id:<40} {exp.task:<12} {exp.agent:<18} {exp.status:<12}"
            )
        
        lines.append("=" * 80)
        lines.append(f"Total experiments: {len(self.experiments)}")
        
        return "\n".join(lines)
    
    def export_csv(self, output_path: Path | str) -> None:
        """Export registry to CSV format."""
        import csv
        
        if not self.experiments:
            return
        
        output_path = Path(output_path)
        
        # Get all field names from the first experiment
        fieldnames = list(list(self.experiments.values())[0].model_dump().keys())
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for exp in self.experiments.values():
                writer.writerow(exp.model_dump())


def extract_metadata_from_run(run_dir: Path) -> ExperimentMetadata | None:
    """
    Extract metadata from a run directory.
    
    Looks for config.json, metrics.json, and infers other metadata.
    
    Args:
        run_dir: Path to the run directory
    
    Returns:
        ExperimentMetadata if sufficient information is found, None otherwise
    """
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    
    # Must have at least config.json to be a valid run
    if not config_path.exists():
        return None
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract basic info from config
    run_id = run_dir.name
    
    # Try to parse date from directory name (format: YYYYMMDD_...)
    created_date = datetime.now().strftime("%Y-%m-%d")
    if len(run_id) >= 8 and run_id[:8].isdigit():
        try:
            created_date = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
        except:
            pass
    
    # Determine task and agent from config or directory name
    task = config.get("task", config.get("env", "ibl_2afc"))
    if "rdm" in run_id.lower() or "macaque" in run_id.lower():
        task = "rdm_macaque"
    elif "ibl" in run_id.lower() or "2afc" in run_id.lower() or "mouse" in run_id.lower():
        task = "ibl_2afc"
    
    # Try to get agent from config (handle both nested and flat structures)
    agent = config.get("agent_type", "unknown")
    if isinstance(config.get("agent"), dict):
        agent_name = config["agent"].get("name", "")
        if "ppo" in agent_name.lower():
            agent = "ppo"
        elif "sticky" in agent_name.lower():
            agent = "sticky_q"
        elif "bayes" in agent_name.lower():
            agent = "bayes_observer"
        elif "ddm" in agent_name.lower() or "hybrid" in agent_name.lower():
            agent = "hybrid_ddm_lstm"
    elif isinstance(config.get("agent"), str):
        agent = config.get("agent")

    # If still unknown, infer from directory name or config structure
    if agent == "unknown":
        if "r_ddm" in run_id.lower() or "rollout_trials" in config:
            agent = "ddm"
        elif "ppo" in run_id.lower():
            agent = "ppo"
        elif "sticky" in run_id.lower() or "stickyq" in run_id.lower():
            agent = "sticky_q"
        elif "bayes" in run_id.lower():
            agent = "bayes_observer"
        elif "hybrid" in run_id.lower() or "wfpt" in run_id.lower():
            agent = "hybrid_ddm_lstm"
        # Check config structure for hybrid agent indicators
        elif "rollout_trials" in config:
            agent = "ddm"
        elif "hidden_size" in config and "loss_weights" in config:
            agent = "hybrid_ddm_lstm"
    
    # Determine status
    status = "experimental"
    if "reference" in run_id.lower():
        status = "reference"
    elif "archive" in str(run_dir).lower():
        status = "archived"
    elif "baseline" in run_id.lower():
        status = "baseline"
    
    # Create base metadata
    metadata = ExperimentMetadata(
        run_id=run_id,
        created_date=created_date,
        task=task,
        agent=agent,
        status=status,
        variant=None,
        seed=config.get("seed"),
        total_trials=None,
        training_time_min=None,
        config_path=str(config_path.relative_to(run_dir.parent.parent)),
        notes=None,
    )
    
    # Add metrics if available
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        metadata.metrics_path = str(metrics_path.relative_to(run_dir.parent.parent))
        
        # Handle both flat and nested metrics structures
        if "metrics" in metrics:
            metrics = metrics["metrics"]
        
        # Extract key metrics
        if "psychometric" in metrics:
            metadata.psychometric_slope = metrics["psychometric"].get("slope")
            metadata.bias = metrics["psychometric"].get("bias")
        
        if "chronometric" in metrics:
            metadata.chronometric_slope = metrics["chronometric"].get("slope_ms_per_unit") or metrics["chronometric"].get("slope")
            metadata.rt_intercept = metrics["chronometric"].get("intercept_ms") or metrics["chronometric"].get("intercept")
            metadata.chronometric_slope_unit = metrics["chronometric"].get("slope_unit")
        
        if "history" in metrics:
            metadata.win_stay_rate = metrics["history"].get("win_stay")
            metadata.lose_shift_rate = metrics["history"].get("lose_shift")
            metadata.sticky_choice_rate = metrics["history"].get("sticky_choice")
        
        if "quality" in metrics and isinstance(metrics["quality"], dict):
            metadata.quality = {str(key): bool(value) for key, value in metrics["quality"].items()}
    
    # Check for other files
    if (run_dir / "trials.ndjson").exists():
        metadata.log_path = str((run_dir / "trials.ndjson").relative_to(run_dir.parent.parent))
    
    if (run_dir / "dashboard.html").exists():
        metadata.dashboard_path = str((run_dir / "dashboard.html").relative_to(run_dir.parent.parent))
    
    return metadata
