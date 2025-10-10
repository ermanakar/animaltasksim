"""Debug script to understand why DDM rollout produces flat RTs."""
import torch
import numpy as np
from agents.hybrid_ddm_lstm import HybridDDMModel

# Load trained model
state_dict = torch.load("runs/rdm_wfpt_minibatch/model.pt", map_location='cpu')
model = HybridDDMModel(feature_dim=7, hidden_size=64, device=torch.device('cpu'), drift_scale=1.0)
model.load_state_dict(state_dict)
model.eval()

print("="*80)
print("DDM ROLLOUT DEBUG")
print("="*80)

# Simulate DDM for different coherences
def simulate_ddm(drift, bound, noise, bias=0.0, dt=0.01, max_steps=120):
    """Replicate the model's DDM simulation."""
    evidence = bias
    sqrt_dt = np.sqrt(dt)
    
    for step in range(max_steps):
        evidence += drift * dt + noise * sqrt_dt * np.random.randn()
        
        if evidence >= bound:
            return 1, step + 1  # Right, num_steps
        elif evidence <= -bound:
            return 0, step + 1  # Left, num_steps
    
    # Timeout
    return (1 if evidence > 0 else 0), max_steps

h = torch.zeros(1, 64)
c = torch.zeros(1, 64)

coherences = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
step_ms = 10

for coh in coherences:
    features = torch.tensor([[coh, abs(coh), 1.0 if coh >= 0 else -1.0, 
                              0.0, 0.0, 0.0, 0.5]])
    out, (h, c) = model(features, (h.detach(), c.detach()))
    
    drift_gain = out['drift_gain'].item()
    drift = drift_gain * coh
    bound = out['bound'].item()
    noise = out['noise'].item()
    bias = out['bias'].item()
    non_decision_ms = out['non_decision_ms'].item()
    
    # Run 100 DDM simulations
    ddm_steps_list = []
    for _ in range(100):
        _, ddm_steps = simulate_ddm(drift, bound, noise, bias, dt=0.01, max_steps=120)
        ddm_steps_list.append(ddm_steps)
    
    # Calculate RT statistics
    ddm_time_ms = np.array(ddm_steps_list) * step_ms
    total_rt_ms = non_decision_ms + ddm_time_ms
    
    # Clip as done in rollout
    total_rt_ms_clipped = np.clip(total_rt_ms, 50, 1200)
    
    mean_ddm_steps = np.mean(ddm_steps_list)
    pct_timeout = (np.array(ddm_steps_list) == 120).mean() * 100
    
    print(f"\nCoherence {coh:.3f}:")
    print(f"  Model outputs:")
    print(f"    drift_gain = {drift_gain:.4f}")
    print(f"    drift      = {drift:.4f}")
    print(f"    bound      = {bound:.4f}")
    print(f"    noise      = {noise:.4f}")
    print(f"    non_dec    = {non_decision_ms:.1f}ms")
    print(f"  DDM simulation (100 trials):")
    print(f"    Mean steps     = {mean_ddm_steps:.1f}")
    print(f"    Timeout rate   = {pct_timeout:.1f}%")
    print(f"    Mean RT (raw)  = {total_rt_ms.mean():.1f}ms ± {total_rt_ms.std():.1f}")
    print(f"    Mean RT (clip) = {total_rt_ms_clipped.mean():.1f}ms")
    print(f"    Min RT         = {total_rt_ms_clipped.min():.1f}ms")
    
    # Key diagnostic: drift signal vs noise
    drift_per_step = drift * 0.01
    noise_per_step = noise * np.sqrt(0.01)
    snr = abs(drift_per_step) / noise_per_step if noise_per_step > 0 else 0
    
    print(f"  Signal-to-noise:")
    print(f"    Drift/step  = {drift_per_step:.6f}")
    print(f"    Noise/step  = {noise_per_step:.6f}")
    print(f"    SNR         = {snr:.6f}")
    
    if snr < 0.01:
        print(f"    ⚠️  SNR too low! Random walk dominates, mostly timeouts")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("If most trials timeout (ddm_steps=120), they get clipped to max RT.")
print("Then commit_step_target = RT_ms / step_ms could be wrong...")
print("Let me check if there's an issue with the commit logic.")
