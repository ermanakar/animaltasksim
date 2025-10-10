#!/usr/bin/env python3
"""Quick evaluation of WFPT-trained model."""

import json
from pathlib import Path

# Wait for training to complete
run_dir = Path("runs/hybrid_wfpt_v1")

if not (run_dir / "trials.ndjson").exists():
    print("Training not complete yet. Waiting...")
    exit(1)

# Evaluate
import subprocess
result = subprocess.run([
    "python", "scripts/evaluate_agent.py",
    "--log", str(run_dir / "trials.ndjson"),
    "--out", str(run_dir / "eval_metrics.json")
], capture_output=True, text=True)

print(result.stdout)

# Load and display key metrics
with open(run_dir / "eval_metrics.json") as f:
    metrics = json.load(f)

print("="*80)
print("WFPT TRAINING RESULTS")
print("="*80)
print()
print("Chronometric (RT-coherence dynamics):")
print(f"  Slope: {metrics['chronometric']['slope_ms_per_unit']:.2f} ms/unit")
print(f"  Target: -655 ms/unit")
print()
print(f"  RT by coherence:")
for coh, rt in metrics['chronometric']['rt_by_level'].items():
    print(f"    {coh}: {rt:.1f}ms")
print()
print("Psychometric (accuracy):")
print(f"  Slope: {metrics['psychometric']['slope']:.3f}")
print(f"  Lapse: {metrics['psychometric']['lapse_low']:.3f}")
print()
print("History effects:")
print(f"  Win-stay: {metrics['history']['win_stay']:.3f}")
print(f"  Lose-shift: {metrics['history']['lose_shift']:.3f}")
print()

# Compare to previous attempts
slope = metrics['chronometric']['slope_ms_per_unit']
target = -655

if abs(slope) > 100:
    print("✅ SUCCESS! Learned evidence-dependent RTs!")
    print(f"   Slope magnitude: {abs(slope):.1f}ms/unit (>{100}ms required)")
elif abs(slope) > 10:
    print("🟡 PARTIAL: Some RT structure learned")
    print(f"   Slope magnitude: {abs(slope):.1f}ms/unit (need {abs(target):.1f}ms)")
else:
    print("❌ FAILED: No RT structure learned")
    print(f"   Slope magnitude: {abs(slope):.1f}ms/unit (<10ms)")

print("="*80)
