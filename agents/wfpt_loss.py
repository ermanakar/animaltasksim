"""
Wiener First Passage Time (WFPT) likelihood loss for DDM training.

This implements the statistically correct objective for DDM parameter estimation:
    Loss = -log p(choice, RT | drift, bound, bias, noise, non_decision)

Based on the Ratcliff/HDDM formulation. Uses infinite series approximation
for the WFPT density, truncated to sufficient terms for numerical stability.

References:
    - Ratcliff & McKoon (2008). The Diffusion Decision Model
    - Wiecki et al. (2013). HDDM: Hierarchical Bayesian estimation of DDM
    - Navarro & Fuss (2009). Fast and accurate calculations for first-passage times
"""
from __future__ import annotations

import torch
import numpy as np


def wfpt_log_likelihood(
    choice: torch.Tensor,
    rt: torch.Tensor,
    drift: torch.Tensor,
    bound: torch.Tensor,
    bias: torch.Tensor,
    noise: torch.Tensor,
    non_decision: torch.Tensor,
    eps: float = 1e-10,
    n_terms: int = 20,
) -> torch.Tensor:
    """
    Compute log-likelihood of (choice, RT) under DDM with WFPT density.
    
    Args:
        choice: Binary choice (0 or 1), shape [batch]
        rt: Reaction time in seconds, shape [batch]
        drift: Drift rate v, shape [batch]
        bound: Boundary separation a > 0, shape [batch]
        bias: Starting point bias z ∈ [0, 1] (fraction of bound), shape [batch]
        noise: Diffusion coefficient s > 0, shape [batch]
        non_decision: Non-decision time Ter ≥ 0, shape [batch]
        eps: Small constant for numerical stability
        n_terms: Number of terms in infinite series approximation
    
    Returns:
        Log-likelihood for each trial, shape [batch]
        More negative = less likely, higher = more likely
    """
    # Convert RT from milliseconds to seconds if needed (assume input is seconds)
    # Subtract non-decision time to get decision time
    decision_time = torch.clamp(rt - non_decision, min=eps)
    
    # Standardize parameters (divide by noise for numerical stability)
    # This is the standard parameterization in DDM literature
    v = drift / (noise + eps)  # drift rate
    a = bound / (noise + eps)  # boundary separation
    z = torch.clamp(bias, min=0.01, max=0.99)  # starting bias (fraction)
    t = decision_time  # decision time
    
    # For upper boundary responses (choice=1), flip sign of drift
    # This is because WFPT is defined for absorption at upper boundary
    v_effective = torch.where(choice == 1, v, -v)
    z_effective = torch.where(choice == 1, z, 1.0 - z)
    
    # Compute WFPT density using infinite series
    # Two formulations: small-time and large-time
    # Use small-time for t*a^2 < 1, large-time otherwise
    
    # Small-time series (good for fast RTs)
    log_p_small = _wfpt_small_time(v_effective, a, z_effective, t, n_terms, eps)
    
    # Large-time series (good for slow RTs)
    log_p_large = _wfpt_large_time(v_effective, a, z_effective, t, n_terms, eps)
    
    # Choose based on t*a^2
    use_small = (t * a ** 2) < 1.0
    log_p = torch.where(use_small, log_p_small, log_p_large)
    
    # Clamp to prevent NaN/Inf
    log_p = torch.clamp(log_p, min=-1000.0, max=10.0)
    
    return log_p


def _wfpt_small_time(
    v: torch.Tensor,
    a: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    n_terms: int,
    eps: float,
) -> torch.Tensor:
    """
    Small-time series for WFPT density.
    Good for t*a^2 < 1 (fast responses).
    
    Formula:
        p(t) = (1/√(2πt³)) * exp(-v*a*z - v²*t/2) * 
               Σ_k (z + 2ka) * exp(-(z + 2ka)²/(2t))
    """
    # Compute base term
    sqrt_2pi_t3 = torch.sqrt(2.0 * np.pi * t ** 3 + eps)
    base = torch.exp(-v * a * z - (v ** 2) * t / 2.0)
    
    # Sum over k terms
    k_values = torch.arange(-n_terms, n_terms + 1, device=v.device, dtype=v.dtype)
    
    # Expand dimensions for broadcasting: [batch, 1] and [2*n_terms+1]
    z_exp = z.unsqueeze(-1)  # [batch, 1]
    a_exp = a.unsqueeze(-1)
    t_exp = t.unsqueeze(-1)
    
    # Compute z + 2*k*a for all k
    z_k = z_exp + 2.0 * k_values * a_exp  # [batch, 2*n_terms+1]
    
    # Compute exponential term
    exp_term = torch.exp(-(z_k ** 2) / (2.0 * t_exp + eps))  # [batch, 2*n_terms+1]
    
    # Sum over k (weighted by z_k)
    summation = torch.sum(z_k * exp_term, dim=-1)  # [batch]
    
    # Combine
    density = (base / (a * sqrt_2pi_t3 + eps)) * summation
    
    # Log density
    log_p = torch.log(torch.clamp(density, min=eps))
    
    return log_p


def _wfpt_large_time(
    v: torch.Tensor,
    a: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    n_terms: int,
    eps: float,
) -> torch.Tensor:
    """
    Large-time series for WFPT density.
    Good for t*a^2 > 1 (slow responses).
    
    Formula:
        p(t) = (π/a²) * exp(-v*a*z - v²*t/2) *
               Σ_k k * sin(π*k*z) * exp(-k²*π²*t/(2*a²))
    """
    # Compute base term
    base = (np.pi / (a ** 2 + eps)) * torch.exp(-v * a * z - (v ** 2) * t / 2.0)
    
    # Sum over k terms (k=1,2,3,...)
    k_values = torch.arange(1, n_terms + 1, device=v.device, dtype=v.dtype)
    
    # Expand dimensions for broadcasting
    z_exp = z.unsqueeze(-1)  # [batch, 1]
    a_exp = a.unsqueeze(-1)
    t_exp = t.unsqueeze(-1)
    
    # Compute k * sin(π*k*z)
    sin_term = k_values * torch.sin(np.pi * k_values * z_exp)  # [batch, n_terms]
    
    # Compute exponential term
    exp_term = torch.exp(-(k_values ** 2) * (np.pi ** 2) * t_exp / (2.0 * a_exp ** 2 + eps))
    
    # Sum over k
    summation = torch.sum(sin_term * exp_term, dim=-1)  # [batch]
    
    # Combine
    density = base * summation
    
    # Log density
    log_p = torch.log(torch.clamp(density, min=eps))
    
    return log_p


def wfpt_loss(
    choice: torch.Tensor,
    rt_ms: torch.Tensor,
    drift: torch.Tensor,
    bound: torch.Tensor,
    bias: torch.Tensor,
    noise: torch.Tensor,
    non_decision_ms: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Negative log-likelihood loss for DDM training.
    
    Args:
        choice: Binary choice (0=left, 1=right), shape [batch]
        rt_ms: Reaction time in milliseconds, shape [batch]
        drift: Drift rate, shape [batch]
        bound: Boundary separation, shape [batch]
        bias: Starting bias, shape [batch] (will be converted to fraction)
        noise: Diffusion coefficient, shape [batch]
        non_decision_ms: Non-decision time in milliseconds, shape [batch]
        weight: Loss weight multiplier
    
    Returns:
        Scalar loss (negative log-likelihood, averaged over batch)
    """
    # Convert RT and non-decision from ms to seconds
    rt_sec = rt_ms / 1000.0
    non_decision_sec = non_decision_ms / 1000.0
    
    # Convert bias from [-1, 1] to [0, 1] (fraction of bound)
    # bias=0 means start at midpoint (0.5)
    # bias>0 means start closer to upper bound
    # bias<0 means start closer to lower bound
    bias_frac = 0.5 + 0.5 * torch.tanh(bias)  # Map to (0, 1)
    
    # Compute log-likelihood
    log_p = wfpt_log_likelihood(
        choice=choice,
        rt=rt_sec,
        drift=drift,
        bound=bound,
        bias=bias_frac,
        noise=noise,
        non_decision=non_decision_sec,
    )
    
    # Negative log-likelihood (we want to maximize likelihood = minimize NLL)
    nll = -log_p
    
    # Average over batch and apply weight
    loss = weight * torch.mean(nll)
    
    return loss


if __name__ == "__main__":
    # Test the WFPT loss
    print("Testing WFPT likelihood loss...")
    print()
    
    # Generate synthetic data
    batch_size = 100
    choice = torch.randint(0, 2, (batch_size,)).float()
    rt_ms = torch.rand(batch_size) * 800 + 200  # 200-1000ms
    
    # DDM parameters (typical values)
    drift = torch.randn(batch_size) * 2.0  # drift ~ N(0, 2)
    bound = torch.ones(batch_size) * 1.0
    bias = torch.zeros(batch_size)
    noise = torch.ones(batch_size) * 1.0
    non_decision_ms = torch.ones(batch_size) * 200.0
    
    # Compute loss
    loss = wfpt_loss(choice, rt_ms, drift, bound, bias, noise, non_decision_ms)
    
    print(f"Batch size: {batch_size}")
    print(f"Loss (NLL): {loss.item():.4f}")
    print()
    
    # Test gradient flow
    drift.requires_grad = True
    bound.requires_grad = True
    
    loss = wfpt_loss(choice, rt_ms, drift, bound, bias, noise, non_decision_ms)
    loss.backward()
    
    print("Gradient check:")
    if drift.grad is not None:
        print(f"  drift.grad: mean={drift.grad.mean():.6f}, std={drift.grad.std():.6f}")
    if bound.grad is not None:
        print(f"  bound.grad: mean={bound.grad.mean():.6f}, std={bound.grad.std():.6f}")
    print()
    print("✅ WFPT loss computes gradients successfully!")
