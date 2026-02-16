"""Test if the rollout is actually computing DDM or using defaults."""
import torch
import numpy as np
from pathlib import Path
from agents.hybrid_ddm_lstm import HybridDDMModel
from envs.rdm_macaque import RDMMacaqueEnv, RDMConfig, AgentMetadata, ACTION_HOLD, ACTION_LEFT, ACTION_RIGHT

# Load trained model
state_dict = torch.load("runs/rdm_wfpt_minibatch/model.pt", map_location='cpu')
model = HybridDDMModel(feature_dim=7, hidden_size=64, device=torch.device('cpu'), drift_scale=1.0)
model.load_state_dict(state_dict)
model.eval()

# Setup minimal environment
env_config = RDMConfig(
    trials_per_episode=10,  # Just 10 trials
    log_path=Path("debug_rollout.ndjson"),
    agent=AgentMetadata(name="debug", version="1.0"),
    seed=42,
    per_step_cost=0.001,
    evidence_gain=0.05,
    momentary_sigma=1.0,
    collapsing_bound=True,
    min_bound_steps=5,
)
env = RDMMacaqueEnv(env_config)
step_ms = env.config.step_ms

# Run one episode with logging
print("="*80)
print("ROLLOUT DEBUG - ONE EPISODE")
print("="*80)

observation, info = env.reset(seed=42)
h, c = model.init_state()
planned_action = ACTION_HOLD
commit_step_target = env.config.min_bound_steps

trial_count = 0
response_phase_count = 0

while True:
    phase = info["phase"]
    
    if phase == "response":
        phase_step = info.get("phase_step", 0)
        
        if phase_step == 0:
            response_phase_count += 1
            print(f"\n--- Trial {trial_count}, Response Phase Start ---")
            print(f"  commit_step_target BEFORE DDM: {commit_step_target}")
            
            # This should happen - compute DDM
            coherence = float(getattr(env, "_signed_coherence", 0.0))
            features = np.array([[coherence, abs(coherence), 1.0 if coherence >= 0 else -1.0,
                                 0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
            x = torch.from_numpy(features)
            out, (h, c) = model(x, (h, c))
            
            drift_gain = out["drift_gain"].item()
            bound = out["bound"].item()
            noise = out["noise"].item()
            bias = out["bias"].item()
            non_decision_ms = out["non_decision_ms"].item()
            drift = drift_gain * coherence
            
            print(f"  Coherence: {coherence:.3f}")
            print(f"  Model: drift={drift:.4f}, bound={bound:.4f}, noise={noise:.4f}")
            
            # Simulate DDM (simplified - just one trial)
            evidence = bias
            sqrt_dt = np.sqrt(0.01)
            for step in range(120):
                evidence += drift * 0.01 + noise * sqrt_dt * np.random.randn()
                if evidence >= bound:
                    ddm_steps = step + 1
                    planned_action = ACTION_RIGHT
                    break
                elif evidence <= -bound:
                    ddm_steps = step + 1
                    planned_action = ACTION_LEFT
                    break
            else:
                ddm_steps = 120
                planned_action = ACTION_RIGHT if evidence > 0 else ACTION_LEFT
            
            ddm_time_ms = ddm_steps * step_ms
            total_rt_ms = non_decision_ms + ddm_time_ms
            total_rt_ms = np.clip(total_rt_ms, step_ms * 5, step_ms * 120)
            commit_step_target = int(total_rt_ms / step_ms)
            commit_step_target = np.clip(commit_step_target, 5, 120)
            
            print(f"  DDM: {ddm_steps} steps → {ddm_time_ms}ms + {non_decision_ms:.0f}ms = {total_rt_ms:.0f}ms")
            print(f"  commit_step_target AFTER DDM: {commit_step_target}")
        
        # Check if we should commit
        current_step = int(info.get("phase_step", 0))  # type: ignore[arg-type]
        if current_step + 1 >= commit_step_target:
            action = planned_action
            if current_step == 0:
                print(f"  Step {current_step}: COMMITTING (current_step+1={current_step+1} >= target={commit_step_target})")
        else:
            action = ACTION_HOLD
            if current_step < 5:  # Only log first few
                print(f"  Step {current_step}: HOLDING (current_step+1={current_step+1} < target={commit_step_target})")
    else:
        action = ACTION_HOLD
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if info["phase"] == "outcome" and info.get("phase_step", 0) == 0:
        actual_rt_ms = float((env._rt_steps or 0) * step_ms)  # noqa: SLF001
        print(f"  OUTCOME: Actual RT = {actual_rt_ms:.0f}ms (expected ~{commit_step_target * step_ms}ms)")
        trial_count += 1
    
    if terminated:
        break

print("\n" + "="*80)
print(f"Completed {trial_count} trials, {response_phase_count} response phases")
print("="*80)
