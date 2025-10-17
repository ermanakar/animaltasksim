#!/usr/bin/env python3
"""
Organize and rename runs/ directory with standardized naming.

This script helps migrate old run directories to a standardized naming convention:
YYYYMMDD_task_agent_variant

It also creates an archive/ directory for old experimental runs and keeps
only key reference and recent runs.

Usage:
    python scripts/organize_runs.py --dry-run  # Preview changes
    python scripts/organize_runs.py             # Apply changes
"""

import json
import shutil
from pathlib import Path

import tyro


def infer_naming_info(run_dir: Path) -> dict[str, str]:
    """
    Infer standardized naming components from run directory.
    
    Returns:
        Dictionary with 'date', 'task', 'agent', 'variant' keys
    """
    name = run_dir.name
    name_lower = name.lower()
    
    # Extract date (if present in format YYYYMMDD)
    date = "20251017"  # Default to today
    if len(name) >= 8 and name[:8].isdigit():
        date = name[:8]
    elif "hybrid_wfpt_curriculum" in name_lower:
        date = "20251012"  # Known from README
    
    # Infer task
    task = "unknown"
    if "ibl" in name_lower or "2afc" in name_lower or "mouse" in name_lower:
        task = "ibl"
    elif "rdm" in name_lower or "macaque" in name_lower:
        task = "rdm"
    
    # Infer agent
    agent = "unknown"
    if "ppo" in name_lower:
        agent = "ppo"
    elif "sticky" in name_lower or "stickyq" in name_lower:
        agent = "stickyq"
    elif "bayes" in name_lower:
        agent = "bayes"
    elif "ddm" in name_lower or "hybrid" in name_lower:
        agent = "hybrid"
    
    # Infer variant from the name
    variant = name
    # Remove date prefix if present
    if variant[:8].isdigit() and len(variant) > 9:
        variant = variant[9:]
    # Remove task/agent prefixes to get variant
    for prefix in ["ibl_", "rdm_", "ppo_", "sticky_", "stickyq_", "bayes_", "hybrid_", "ddm_"]:
        if variant.lower().startswith(prefix):
            variant = variant[len(prefix):]
    
    return {
        "date": date,
        "task": task,
        "agent": agent,
        "variant": variant,
    }


def suggest_new_name(run_dir: Path) -> str:
    """Generate standardized name for run directory."""
    info = infer_naming_info(run_dir)
    
    # Format: YYYYMMDD_task_agent_variant
    parts = [info["date"], info["task"], info["agent"]]
    if info["variant"] and info["variant"] != run_dir.name:
        parts.append(info["variant"])
    
    return "_".join(parts)


def should_archive(run_dir: Path) -> bool:
    """
    Determine if a run should be archived.
    
    Criteria:
    - Contains 'attempt' in name (experimental iteration)
    - Contains 'sweep' in name (hyperparameter search)
    - Contains 'v1', 'v2' in name (old versions)
    - Is in the 'archive' directory already
    - Doesn't have critical files (config.json, metrics.json)
    """
    name_lower = run_dir.name.lower()
    
    # Already archived
    if "archive" in str(run_dir):
        return False  # Don't re-archive
    
    # Definitely archive
    archive_patterns = [
        "attempt",
        "sweep",
        "_v1",
        "_v2",
        "annealed",
        "focused",
        "guarded",
        "finetune",
        "supervision",
        "calibration_logs",
        "drift_scale_",
        "final_sweep_",
        "history_embedding",
        "history_finetune",
        "history_supervision",
        "rt_calibration",
        "rt_weighted",
    ]
    
    if any(pattern in name_lower for pattern in archive_patterns):
        return True
    
    # Keep reference runs and final results
    keep_patterns = [
        "reference",
        "hybrid_wfpt_curriculum",
        "latest",
        "final",
    ]
    
    if any(pattern in name_lower for pattern in keep_patterns):
        return False
    
    # Check for essential files
    has_config = (run_dir / "config.json").exists()
    has_metrics = (run_dir / "metrics.json").exists()
    
    # Archive if missing both
    if not has_config and not has_metrics:
        return True
    
    return False


def organize_runs(
    runs_dir: Path = Path("runs"),
    dry_run: bool = True,
) -> None:
    """
    Organize runs directory with standardized naming.
    
    Args:
        runs_dir: Directory containing runs
        dry_run: If True, only preview changes without applying them
    """
    # Create archive directory
    archive_dir = runs_dir / "archive"
    
    if not dry_run:
        archive_dir.mkdir(exist_ok=True)
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Organizing runs in {runs_dir}...\n")
    
    actions = []
    
    # Scan all run directories
    for item in sorted(runs_dir.iterdir()):
        if not item.is_dir():
            continue
        
        # Skip special directories
        if item.name in ["archive", ".DS_Store"] or item.name.startswith("."):
            continue
        
        # Determine action
        if should_archive(item):
            action = "ARCHIVE"
            target = archive_dir / item.name
        else:
            # Check if rename is needed
            suggested_name = suggest_new_name(item)
            if suggested_name != item.name and not item.name.startswith("20"):
                action = "RENAME"
                target = runs_dir / suggested_name
            else:
                action = "KEEP"
                target = item
        
        actions.append({
            "source": item,
            "target": target,
            "action": action,
        })
    
    # Print summary
    archive_count = sum(1 for a in actions if a["action"] == "ARCHIVE")
    rename_count = sum(1 for a in actions if a["action"] == "RENAME")
    keep_count = sum(1 for a in actions if a["action"] == "KEEP")
    
    print(f"Summary:")
    print(f"  Archive: {archive_count}")
    print(f"  Rename: {rename_count}")
    print(f"  Keep: {keep_count}")
    print()
    
    # Show actions
    for action_info in actions:
        action = action_info["action"]
        source = action_info["source"]
        target = action_info["target"]
        
        if action == "KEEP":
            continue
        
        if action == "ARCHIVE":
            print(f"[ARCHIVE] {source.name} → archive/{target.name}")
        elif action == "RENAME":
            print(f"[RENAME]  {source.name} → {target.name}")
    
    # Apply changes
    if not dry_run:
        print("\nApplying changes...")
        
        for action_info in actions:
            action = action_info["action"]
            source = action_info["source"]
            target = action_info["target"]
            
            if action == "KEEP":
                continue
            
            try:
                if target.exists():
                    print(f"  ⚠️  Target exists: {target.name}, skipping")
                    continue
                
                shutil.move(str(source), str(target))
                print(f"  ✅ {action}: {source.name}")
            
            except Exception as e:
                print(f"  ❌ Error: {source.name} - {e}")
        
        print("\nDone!")
    else:
        print("\n[DRY RUN] Use --no-dry-run to apply these changes.")


if __name__ == "__main__":
    tyro.cli(organize_runs)
