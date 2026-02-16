"""Generate synthetic DDM data for supervised pretraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SyntheticDDMConfig:
    """Configuration for synthetic DDM data generation."""
    
    n_trials: int = 10000
    coherences: list[float] | None = None  # Will use default if None
    drift_rate_range: tuple[float, float] = (3.0, 8.0)  # Multiplier for coherence
    bound_range: tuple[float, float] = (0.8, 2.0)
    noise_range: tuple[float, float] = (0.8, 1.2)
    non_decision_range: tuple[float, float] = (150.0, 200.0)  # ms
    bias_range: tuple[float, float] = (-0.1, 0.1)
    step_ms: float = 10.0
    max_steps: int = 120
    seed: int = 42
    
    def __post_init__(self):
        if self.coherences is None:
            self.coherences = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]


def simulate_ddm_trial(
    coherence: float,
    drift_rate: float,
    bound: float,
    noise: float,
    non_decision_ms: float,
    bias: float,
    step_ms: float = 10.0,
    max_steps: int = 120,
) -> tuple[int, float, int]:
    """
    Simulate single DDM trial with Euler-Maruyama integration.
    
    Returns:
        action (0=left, 1=right), rt_ms, steps_taken
    """
    evidence = bias
    dt = step_ms / 1000.0
    sqrt_dt = np.sqrt(dt)
    
    # Drift is drift_rate * coherence
    drift = drift_rate * coherence
    
    for step in range(max_steps):
        evidence += drift * dt + noise * sqrt_dt * np.random.randn()
        
        if abs(evidence) >= bound:
            action = 1 if evidence > 0 else 0
            rt_ms = non_decision_ms + (step + 1) * step_ms
            return action, rt_ms, step + 1
    
    # Timeout - random choice
    action = np.random.choice([0, 1])
    rt_ms = non_decision_ms + max_steps * step_ms
    return action, rt_ms, max_steps


def generate_synthetic_dataset(config: SyntheticDDMConfig) -> list[dict]:
    """Generate synthetic DDM dataset with known parameters."""
    np.random.seed(config.seed)
    
    dataset = []
    coherence_list = config.coherences if config.coherences is not None else [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
    
    for trial_idx in range(config.n_trials):
        # Sample parameters
        coherence = np.random.choice(coherence_list)
        # Add sign
        signed_coherence = coherence * np.random.choice([-1, 1])
        
        drift_rate = np.random.uniform(*config.drift_rate_range)
        bound = np.random.uniform(*config.bound_range)
        noise = np.random.uniform(*config.noise_range)
        non_decision_ms = np.random.uniform(*config.non_decision_range)
        bias = np.random.uniform(*config.bias_range)
        
        # Simulate trial
        action, rt_ms, steps = simulate_ddm_trial(
            signed_coherence,
            drift_rate,
            bound,
            noise,
            non_decision_ms,
            bias,
            config.step_ms,
            config.max_steps,
        )
        
        # Calculate drift_gain (what the model should predict)
        # drift_gain is the coefficient: drift = drift_gain * coherence
        drift_gain = drift_rate
        
        # Store trial data
        trial = {
            "trial_index": trial_idx,
            "coherence": float(signed_coherence),
            "abs_coherence": float(abs(signed_coherence)),
            "sign": float(np.sign(signed_coherence)) if signed_coherence != 0 else 0.0,
            "action": action,
            "rt_ms": float(rt_ms),
            "steps": steps,
            # Ground truth parameters (for supervised learning)
            "true_drift_gain": float(drift_gain),
            "true_bound": float(bound),
            "true_noise": float(noise),
            "true_non_decision_ms": float(non_decision_ms),
            "true_bias": float(bias),
        }
        dataset.append(trial)
    
    return dataset


def save_synthetic_dataset(dataset: list[dict], path: Path) -> None:
    """Save synthetic dataset to NDJSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for trial in dataset:
            # Convert NumPy types to Python native types for JSON serialization
            json_trial = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                         for k, v in trial.items()}
            f.write(json.dumps(json_trial) + "\n")


def load_synthetic_dataset(path: Path) -> list[dict]:
    """Load synthetic dataset from NDJSON."""
    dataset = []
    with open(path) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def analyze_synthetic_dataset(dataset: list[dict]) -> None:
    """Print statistics about synthetic dataset."""
    from scipy import stats as scipy_stats
    
    rts = np.array([t["rt_ms"] for t in dataset])
    coherences = np.array([t["coherence"] for t in dataset])
    abs_coh = np.abs(coherences)
    drift_gains = np.array([t["true_drift_gain"] for t in dataset])
    
    print("\n" + "="*80)
    print("SYNTHETIC DATASET ANALYSIS")
    print("="*80)
    print(f"\nTotal trials: {len(dataset)}")
    print("\nRT statistics:")
    print(f"  Mean: {rts.mean():.1f}ms")
    print(f"  Std: {rts.std():.1f}ms")
    print(f"  Range: [{rts.min():.0f}, {rts.max():.0f}]ms")
    
    print(f"\nDrift gain range: [{drift_gains.min():.2f}, {drift_gains.max():.2f}]")
    
    print("\nRT by coherence:")
    for coh in sorted(np.unique(abs_coh)):
        mask = abs_coh == coh
        coh_rts = rts[mask]
        print(f"  Coh {coh:5.3f}: mean={coh_rts.mean():6.1f}ms, std={coh_rts.std():5.1f}ms, n={mask.sum()}")
    
    # RT-coherence regression
    linreg = scipy_stats.linregress(abs_coh, rts)
    slope = linreg.slope  # type: ignore
    r = linreg.rvalue  # type: ignore
    r2 = r ** 2
    
    print("\nRT-Coherence relationship:")
    print(f"  Slope: {slope:.1f} ms/unit")
    print(f"  R²: {r2:.4f}")
    print(f"  RT difference (hard-easy): {rts[abs_coh < 0.05].mean() - rts[abs_coh > 0.5].mean():.1f}ms")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Generate synthetic dataset
    config = SyntheticDDMConfig(
        n_trials=10000,
        drift_rate_range=(4.0, 8.0),  # Strong drift for clear RT-coherence relationship
        seed=42,
    )
    
    print("Generating synthetic DDM dataset...")
    dataset = generate_synthetic_dataset(config)
    
    # Save to file
    output_path = Path("data/synthetic/ddm_pretraining.ndjson")
    save_synthetic_dataset(dataset, output_path)
    
    # Analyze
    analyze_synthetic_dataset(dataset)
