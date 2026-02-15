# Hybrid DDM + LSTM Agent Design

> **Note (Feb 2026):** This is the original design document from October 2025. The implementation has since evolved — notably, the separate behavioral cloning / RT distribution / history regularization losses described in §2.1 were replaced by a unified WFPT (Wiener First Passage Time) likelihood loss with curriculum learning. See `FINDINGS.md` for current results and `agents/hybrid_ddm_lstm.py` for the current implementation.

**Objective:** produce an agent that reproduces animal-like choice, reaction time, and history fingerprints on the macaque RDM task (and eventually other tasks) by combining mechanistic evidence accumulation with learned history-dependent control.

---

## 1. Architectural Overview

### 1.1 Components

- **Recurrent Controller (LSTM/GRU):** ingests per-trial summaries (stimulus coherence, previous choice, previous outcome, block prior if exposed, elapsed RT) and outputs latent state `h_t` capturing history effects beyond first-order dependencies.
- **Parameter Heads:** linear layers projecting `h_t` (plus current stimulus features) into DDM parameters:
  - `drift_gain_t` (signed accumulation rate)
  - `bound_t` (decision thresholds)
  - `bias_t` (starting point or constant evidence offset)
  - optional `non_decision_ms_t` for per-trial latency.
- **Differentiable DDM Simulator:** integrates evidence using Euler-Maruyama steps with Gaussian noise to generate simulated reaction times and choices. Simulation is differentiable via reparameterisation and stopped gradient for discrete choice sample.
- **Action Selection Wrapper:** maps simulated choice to Gym actions (`left`, `right`, `hold`) and tracks RT in env steps so the env sees consistent behaviour.

### 1.2 Forward Pass (per trial)

1. At the start of `response` phase, assemble feature vector:
   ```python
   x_t = [coherence, sign(coherence), abs(coherence), prev_choice, prev_reward, prev_correct, trial_index_norm]
   ```
2. Pass `x_t` through LSTM cell with hidden state `h_{t-1}` to obtain `(h_t, c_t)`.
3. Compute DDM parameters:
   ```python
   drift_gain_t = softplus(W_drift @ h_t + b_drift) * sign(coherence)
   bound_t = softplus(W_bound @ h_t + b_bound) + eps
   bias_t = tanh(W_bias @ h_t + b_bias)
   ```
4. Simulate evidence accumulation until |evidence| ≥ bound or `max_steps` reached; convert simulated steps to RT in ms.
5. Return action (`left` or `right`) plus RT metadata to training loop.

---

## 2. Training Strategy

### 2.1 Multi-Objective Loss

Total loss combines four terms with tunable weights `α, β, γ, δ`:

- **Task Reward Loss (`L_rl`)**: negative of cumulative reward across episode (REINFORCE/actor-critic with advantage baseline). Keeps agent on-policy and maintains high task performance.
- **Behavioral Cloning (`L_bc`)**: cross-entropy between agent choice probabilities and reference animal choices for matched stimuli.
- **RT Distribution Loss (`L_rt`)**: mean-squared error on mean and standard deviation of RT, plus Wasserstein-1 distance between agent and animal RT samples using differentiable approximation.
- **History Regularization (`L_hist`)**: KL divergence between agent’s empirical win-stay/lose-shift rates and reference values (computed per batch).

### 2.2 Data Usage

- Load reference `.ndjson` logs, precompute per-trial targets (choice, RT, history metrics) and sample batches during training for BC/RT/History losses.
- Alternate between on-policy rollouts in env and supervised steps on reference data to stabilise training (interleave or use replay buffer mixing agent/animal trials).

### 2.3 Optimisation

- Use Adam with gradient clipping (norm ≤ 5) to stabilise recurrent training.
- Temperature parameter controls stochastic action sampling; anneal slowly to avoid deterministic collapse.
- Early stopping criterion based on validation metrics (held-out animal sessions).

---

## 3. Integration Points

### 3.1 Code Layout

```
agents/
├─ hybrid_ddm_lstm.py       # Agent module (controller + simulator + training loop)
├─ losses.py                # Shared loss utilities (BC, RT, history, Wasserstein)
└─ __init__.py              # Export new classes

scripts/
└─ train_hybrid.py          # CLI entry point, mirrors train_agent.py pattern
```

### 3.2 Configuration

Use dataclasses mirroring existing patterns:

```python
@dataclass
class HybridDDMConfig:
    trials_per_episode: int = 400
    episodes: int = 50
    seed: int = 1234
    hidden_size: int = 64
    max_steps: int = 120
    learning_rate: float = 1e-3
    loss_weights: LossWeights = LossWeights()
    reference_log: Path
    output_dir: Path
```

Persist configs via `config.json` in run directory and record training timestamps, seeds, and loss weights.

### 3.3 Schema Extensions

Add optional keys to trial logs (maintain backward compatibility via defaults):
- `model_state`: truncated summary (e.g., initial bias, drift, bound) per trial.
- `sim_rt_ms`: internally simulated RT used for action triggering.

Schema changes require updates to `eval/schema_validator.py` and new tests verifying presence/absence of optional fields.

---

## 4. Evaluation & Validation

1. **Unit Tests**: ensure simulator produces plausible RT distributions for fixed parameters; verify loss utilities (BC, RT, history) compute expected values on toy data.
2. **Integration Tests**: short training run (2 episodes, tiny network) completes <1 min and logs schema-compliant trials.
3. **Metrics**: extend `eval/metrics.py` to include RT quantiles and choice autocorrelation; test using synthetic logs.
4. **Cross-Validation**: add helper to split reference logs into train/validation sessions and report generalisation gap.

---

## 5. Risks & Mitigations

- **Simulation stability:** clamp bounds and drift to avoid runaway evidence; use adaptive step size when coherence is tiny.
- **Gradient variance:** mix supervised and RL batches; optionally use straight-through estimator for discrete choice.
- **Schema drift:** keep new log keys optional with defaults so legacy logs remain valid.
- **Runtime:** provide compact hyperparameters for CI (hidden_size=16, episodes=2) and guard main training configs with documentation.

---

## 6. Deliverables for Phase 1

1. Implement `HybridDDMController` and differentiable simulator.
2. Add training CLI with config persistence, seeding, and `.ndjson` logging.
3. Write unit tests for simulator and losses; add smoke test for CLI.
4. Document usage in README (`Quickstart` + new section) and FINDINGS addendum describing architecture motivation.

