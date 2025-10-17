#!/usr/bin/env python3
"""
Clean up incomplete and redundant runs based on data analysis.

This script removes runs that:
- Are intermediate attempts without results (metrics/dashboard)
- Have been superseded by later versions
- Are incomplete (missing critical files)

Usage:
    python scripts/cleanup_runs.py --dry-run   # Preview deletions
    python scripts/cleanup_runs.py             # Execute deletions
"""

import shutil
from pathlib import Path

import tyro

from animaltasksim.registry import ExperimentRegistry


def identify_deletable_runs(runs_dir: Path = Path("runs")) -> list[tuple[str, str]]:
    """
    Identify runs that can be safely deleted.
    
    Returns:
        List of (run_id, reason) tuples
    """
    deletable = []
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name in ["archive", ".DS_Store"]:
            continue
        
        run_id = run_dir.name
        
        # Check what files exist
        has_config = (run_dir / "config.json").exists()
        has_trials = (run_dir / "trials.ndjson").exists()
        has_metrics = (run_dir / "metrics.json").exists()
        has_dashboard = (run_dir / "dashboard.html").exists()
        has_model = any([
            (run_dir / "model.pt").exists(),
            (run_dir / "model_phase1_wfpt_only.pt").exists(),
            (run_dir / "model_phase2_full_balance.pt").exists(),
        ])
        
        # Criteria for deletion
        should_delete = False
        reason = ""
        
        # Delete if intermediate attempt/version without results
        if ("attempt" in run_id or "_v1" in run_id or "_v2" in run_id) and not has_dashboard and not has_metrics:
            should_delete = True
            reason = "Intermediate attempt/version without results"
        
        # Delete if only has config (incomplete)
        if has_config and not any([has_trials, has_metrics, has_dashboard, has_model]):
            should_delete = True
            reason = "Incomplete (config only)"
        
        # Never delete these important runs
        important = [
            "hybrid_wfpt_curriculum",
            "hybrid_wfpt_curriculum_timecost",
            "hybrid_wfpt_curriculum_timecost_soft_rt",
            "ibl_stickyq_latency",
            "rdm_ppo_latest",
            "rdm_wfpt_regularized",
            "rdm_hybrid_curriculum",
            "final_history_embedding_agent",
        ]
        
        if run_id in important:
            should_delete = False
        
        if should_delete:
            deletable.append((run_id, reason))
    
    return deletable


def cleanup_runs(
    runs_dir: Path = Path("runs"),
    dry_run: bool = True,
) -> None:
    """
    Clean up incomplete and redundant runs.
    
    Args:
        runs_dir: Directory containing runs
        dry_run: If True, only preview deletions without executing
    """
    print(f"{'[DRY RUN] ' if dry_run else ''}Analyzing runs for cleanup...\n")
    
    deletable = identify_deletable_runs(runs_dir)
    
    if not deletable:
        print("✅ No runs identified for deletion. All runs are complete or important.")
        return
    
    # Show what will be deleted
    print(f"Found {len(deletable)} runs to delete:\n")
    
    total_size = 0
    for run_id, reason in deletable:
        run_dir = runs_dir / run_id
        
        # Calculate size
        size = 0
        if run_dir.exists():
            size = sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())
            total_size += size
        
        size_mb = size / 1024 / 1024
        print(f"  ❌ {run_id:<45} ({size_mb:>5.1f} MB) - {reason}")
    
    print(f"\nTotal space to free: {total_size / 1024 / 1024:.1f} MB")
    
    # Execute deletions
    if not dry_run:
        print("\n⚠️  Proceeding with deletions...")
        
        # Load registry to update it
        registry_path = runs_dir / "registry.json"
        registry = ExperimentRegistry(registry_path)
        
        deleted = 0
        failed = 0
        
        for run_id, reason in deletable:
            run_dir = runs_dir / run_id
            
            try:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    print(f"  ✅ Deleted: {run_id}")
                    deleted += 1
                
                # Remove from registry
                if run_id in registry.experiments:
                    registry.delete(run_id)
            
            except Exception as e:
                print(f"  ❌ Failed to delete {run_id}: {e}")
                failed += 1
        
        print(f"\n✅ Cleanup complete!")
        print(f"   Deleted: {deleted}")
        print(f"   Failed: {failed}")
        print(f"   Space freed: {total_size / 1024 / 1024:.1f} MB")
        print(f"\n📊 Registry updated: {len(registry.experiments)} experiments remaining")
    
    else:
        print(f"\n[DRY RUN] Use --no-dry-run to execute deletions.")


if __name__ == "__main__":
    tyro.cli(cleanup_runs)
