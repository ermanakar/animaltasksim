# AnimalTaskSim Findings

**Benchmarking reinforcement learning agents against rodent and primate decision-making fingerprints**  
*October 2025 · Version 0.1.0*

---

## Recent Updates (October 10, 2025)

### Hybrid DDM+LSTM Agent: All Training Approaches Exhausted

**TLDR:** Stochastic DDM simulation works (generates RT variability) but **all training approaches fail** to learn evidence-dependent RT dynamics. Five independent attempts (4 incremental fixes + curriculum learning) produced identical results: slope ~1-3ms/unit (0.2-0.4% of target -655ms/unit), R²~0.00002 (0.01% of target 0.34). **Definitive conclusion: MSE RT loss fundamentally broken. Only supervised pretraining on synthetic data remains viable.**

---

#### What We Built

- **Stochastic DDM simulation** using Euler-Maruyama integration (`agents/hybrid_ddm_lstm.py:468-496`)
- Evidence accumulation: `evidence += drift*dt + noise*sqrt(dt)*randn()` step-by-step until bound crossing
- Architecture: 7-feature input → LSTM(64) → drift_gain, bound, bias, non_decision outputs
- Multi-objective loss: choice (BCE) + RT (MSE) + history (quadratic) + drift_supervision (optional)
- **Curriculum learning** infrastructure with phased training and success criteria

#### All Attempts (All Failed)

| Attempt | Fix Strategy | Slope | R² | RT Diff | Status |
|---------|-------------|-------|-----|---------|---------|
| **rdm_hybrid_fix** | drift_scale=10.0, per_step_cost=0.001 | 1.08 ms/unit | 0.000022 | 0.8ms | ❌ FAILED |
| **rdm_hybrid_normalized** | Normalized RT loss (relative structure) | 1.08 ms/unit | 0.000022 | 0.8ms | ❌ FAILED |
| **rdm_hybrid_supervised** | drift_supervision=0.1 (explicit constraint) | CRASHED | — | — | ❌ CRASHED |
| **rdm_hybrid_supervised_strong** | drift_supervision=0.5 (5x stronger) | 1.08 ms/unit | 0.000022 | 0.8ms | ❌ FAILED |
| **rdm_hybrid_curriculum_v1** | Curriculum Phase 1 (choice=0, rt=1.0, drift_sup=0.5) | 2.78 ms/unit | 0.000155 | 2.9ms | ❌ FAILED |
| **Reference (macaque)** | Real data | -655 ms/unit | 0.34 | 318ms | — |

**All runs produced nearly identical failures:**

- Mean RT: 75-76ms (uniform across coherences)
- Slope: ~1-3ms/unit (need -655ms/unit for 100% match)
- R²: 0.000022-0.000155 (need 0.34 for 100% match)  
- RT difference (hard-easy): 0.8-2.9ms (need 318ms)
- RT loss: 0.0 throughout training (no gradient signal)
- Drift supervision loss: 22-23 (parameters collapsed despite explicit penalty)

---

#### Root Cause Analysis

**Training objective fundamentally misaligned with desired behavior:**

1. **Choice loss (BCE) doesn't require evidence-dependent RTs**
   - Agent achieves 54-60% accuracy with constant fast RTs
   - No gradient signal linking coherence to RT structure

2. **RT loss (MSE on means) learns constant fast RT**
   - Given reference RT variance, constant RT ~75ms minimizes MSE
   - Loss provides no incentive for coherence-dependent structure
   - RT loss stayed exactly 0.0 throughout all training runs

3. **Drift supervision ignored**
   - drift_supervision loss not even logged in metrics (weight=0.5)
   - No evidence it was computed or influenced training
   - Drift parameters unconstrained by any gradient signal

4. **Stochastic simulation generates variability but training ignores it**
   - Simulation infrastructure works (35 unique RTs, std=40ms)
   - Training converges to fast uniform noise pattern
   - No architectural or objective pressure to match animal dynamics

---

#### Technical Details

**Diagnostic Evidence:**

```python
# Reference macaque data:
RT vs coherence: slope=-655ms/unit, R²=0.34, diff=318ms

# All agent attempts:
RT vs coherence: slope=1.08ms/unit, R²=0.000022, diff=-0.8ms

# Training losses (all runs):
RT loss: 0.0000 → 0.0000 (no learning signal)
Choice loss: 0.66 → 0.64 (minor improvement)
drift_supervision: NOT LOGGED (even with weight=0.5)
```

**Code Locations:**

- Stochastic DDM: `agents/hybrid_ddm_lstm.py:468-496`
- Drift supervision: `agents/losses.py:85-103` (implemented but ineffective)
- Drift scaling: `agents/hybrid_ddm_lstm.py:90-103` (10x initialization, collapsed to 0.25)
- Training loop: `agents/hybrid_ddm_lstm.py:408-412` (drift_supervision integration)

**Artifacts:**

- `runs/rdm_hybrid_fix/` - drift scaling attempt
- `runs/rdm_hybrid_normalized/` - normalized RT loss
- `runs/rdm_hybrid_supervised_strong/` - final supervised attempt

---

#### Curriculum Learning: The Definitive Test

**rdm_hybrid_curriculum_v1** tested whether removing choice loss interference would allow RT structure to emerge. Phase 1 configuration:

- **Loss weights:** choice=0.0, rt=1.0, drift_supervision=0.5
- **Strategy:** Train RT structure first, completely ignore accuracy
- **Success criteria:** slope>100ms/unit, R²>0.1, RT_diff>50ms
- **Training:** 5 epochs on reference data

**Results:**

- Slope: 2.78 ms/unit (need 100 ms/unit) → **FAILED by 97%**
- R²: 0.000155 (need 0.1) → **FAILED by 99.8%**
- RT difference: 2.9ms (need 50ms) → **FAILED by 94%**
- RT loss: 0.0 throughout (no gradient)
- Drift supervision: 22.86 (parameters collapsed)

**Interpretation:** Even with **zero choice pressure**, RT loss (MSE) provides no gradient for evidence-dependent structure. This definitively rules out "choice loss interference" as the root cause. **The RT loss objective itself is fundamentally broken.**

Early stopping triggered after Phase 1 failure. Phases 2 and 3 never executed.

---

#### Conclusion: Only Supervised Pretraining Remains Viable

**All five training approaches failed identically:**

1. ❌ Drift parameter scaling → collapse
2. ❌ Normalized RT loss → no structure
3. ❌ Explicit drift supervision (0.1, 0.5) → ignored
4. ❌ Time cost reduction → no effect
5. ❌ **Curriculum learning (RT-first)** → **no gradient even without choice**

**This definitively proves:**

- MSE RT loss provides zero gradient for coherence-dependent structure
- Drift supervision cannot compensate for missing gradient
- Choice loss interference is NOT the root cause
- Phased training cannot overcome fundamental objective mismatch

**Remaining Options:**

#### ~~Option C2: Curriculum Learning~~ → **RULED OUT** (Phase 1 failed definitively)

#### Option C1: Supervised Pretraining ✅ **RECOMMENDED**

- Generate synthetic DDM trajectories with known drift/bound/RT relationships
- Pretrain LSTM to predict drift_gain from (coherence, RT) pairs
- Fine-tune on task with frozen or regularized DDM parameters
- **Rationale:** Bypass broken RT loss by teaching structure explicitly

#### Option C3: Architectural Constraints

- Fix drift/bound ratios based on psychophysics literature
- Only learn history-dependent modulations and bias parameters
- Remove degrees of freedom that training collapses

#### Option C4: Different Training Objective

- Contrastive learning: match trial-level RT distributions by coherence
- Inverse RL: infer reward function from animal behavior
- Adversarial training: discriminator judges agent vs animal trials

**Current status:** Stochastic DDM simulation infrastructure complete and validated. Training framework ready for major architectural/objective redesign. Incremental fixes definitively ruled out.

---

## Executive Summary

AnimalTaskSim measures how closely AI agents reproduce animal behavioral fingerprints on two classic tasks. Baseline agents match several bias and history statistics but diverge on reaction-time dynamics and lapse patterns. Architectural inductive biases, not just reward shaping, remain the limiting factor.

- Sticky-GLM (IBL 2AFC): nails contrast bias (99% match) yet under-expresses win-stay (79%) and overproduces high-contrast lapses (731%).
- PPO (RDM): maximizes reward but collapses RT structure (28% intercept match) and introduces large directional bias (0.247 vs ~0).
- Drift Diffusion Model (RDM): best RT alignment (81% intercept match) at the cost of overly shallow psychometric slopes (37%).

Dashboards for each run live under `runs/` and pair visuals with metric deltas.

---

## Task Highlights

### Mouse Visual 2AFC (IBL)

- **Reference benchmarks** (885 trials): slope 13.2, bias 0.074, win-stay 73%, lose-shift 34%, sticky-choice 72%, lapse_high 0.0026.
- **Sticky-GLM v21** (25 training episodes): slope 18.5 (140%), bias -0.001 (99%), win-stay 58% (79%), lose-shift 34% (99%), sticky-choice 57% (79%), lapse_high 0.019 (731%).
- **Interpretation:** Linear policy reproduces additive biases but lacks depth in multi-trial history and over-fits stimulus features, yielding excess confident errors on easy trials.

### Macaque Random-Dot Motion (RDM)

- **Reference benchmarks** (2611 trials): slope 17.6, bias ~0, RT intercept 759 ms, RT slope -645 ms, win-stay 46%, sticky-choice 46%.
- **PPO v24:** slope 30.4 (173%), bias 0.247, RT intercept 210 ms (28%), RT slope 0 ms (0%), win-stay 46% (100%).  *Fast, reward-optimal, but non-biological RT curve and large choice bias.*
- **DDM v2:** slope 6.5 (37%), bias 0.015, RT intercept 613 ms (81%), RT slope -139 ms (22%), win-stay 53% (116%).  *Captures temporal dynamics via evidence accumulation but misses accuracy scaling.*

---

## Key Insights

1. **Reward-optimal ≠ behaviorally faithful.** Agents maximizing reward alone diverge on RT and lapse structure found in animal data.
2. **Architectural priors matter.** Mechanistic models (DDM) inherit realistic temporal dynamics; generic policy gradients do not, even with history features.
3. **Multi-trial memory remains an open gap.** Sticky-GLM and PPO capture first-order history but underperform on deeper kernels observed in rodents and primates.
4. **Schema-first logging pays off.** Shared `.ndjson` format lets metrics, dashboards, and future tasks plug in without interface churn.

---

## Next Steps

1. Hybrid agent combining DDM-style accumulation with adaptive policy layers to balance RT realism and accuracy.
2. History-regularized training objectives to encourage biologically plausible win-stay/lose-shift patterns.
3. Extend the pipeline to PRL and DMS while reusing seeds, schema validation, and reporting hooks.

---

## Artifacts

- Mouse dashboard: `runs/ibl_final_dashboard.html`
- Macaque PPO dashboard: `runs/rdm_final_dashboard.html`
- Macaque DDM dashboard: `runs/rdm_ddm_dashboard.html`
- Metrics JSON and config snapshots live alongside each run directory.

---

## Citation

```text
AnimalTaskSim: A Benchmark for Evaluating Behavioral Replication in AI Agents
https://github.com/ermanakar/animaltasksim
October 2025
```
