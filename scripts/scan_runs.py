#!/usr/bin/env python3
"""
Scan runs/ directory and populate the experiment registry.

This script walks through the runs/ directory, extracts metadata from each
run, and adds it to the central registry.json file.

Usage:
    python scripts/scan_runs.py [--registry PATH] [--overwrite]
"""

from pathlib import Path

import tyro

from animaltasksim.registry import ExperimentRegistry, extract_metadata_from_run


def scan_runs(
    runs_dir: Path = Path("runs"),
    registry_path: Path = Path("runs/registry.json"),
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Scan runs directory and populate the experiment registry.
    
    Args:
        runs_dir: Directory containing experiment runs
        registry_path: Path to registry JSON file
        overwrite: If True, overwrite existing entries
        verbose: If True, print progress
    """
    registry = ExperimentRegistry(registry_path)
    
    if verbose:
        print(f"Scanning {runs_dir}...")
    
    scanned = 0
    added = 0
    skipped = 0
    failed = 0
    
    # Recursively find all directories that might be runs
    for item in runs_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Skip special directories
        if item.name.startswith(".") or item.name == "archive":
            continue
        
        scanned += 1
        
        # Try to extract metadata
        try:
            metadata = extract_metadata_from_run(item)
            
            if metadata is None:
                if verbose:
                    print(f"  ⚠️  {item.name}: No config.json found, skipping")
                skipped += 1
                continue
            
            # Check if already in registry
            if metadata.run_id in registry.experiments and not overwrite:
                if verbose:
                    print(f"  ⏩ {item.name}: Already in registry, skipping (use --overwrite to update)")
                skipped += 1
                continue
            
            # Add to registry
            registry.add(metadata, overwrite=overwrite)
            added += 1
            
            if verbose:
                print(f"  ✅ {item.name}: Added to registry")
                print(f"      Task: {metadata.task}, Agent: {metadata.agent}, Status: {metadata.status}")
        
        except Exception as e:
            failed += 1
            if verbose:
                print(f"  ❌ {item.name}: Error - {e}")
    
    # Summary
    if verbose:
        print("\nScan complete!")
        print(f"  Scanned: {scanned}")
        print(f"  Added: {added}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")
        print(f"\nRegistry saved to: {registry_path}")
        print(f"Total experiments in registry: {len(registry.experiments)}")


if __name__ == "__main__":
    tyro.cli(scan_runs)
