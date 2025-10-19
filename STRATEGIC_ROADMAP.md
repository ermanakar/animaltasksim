# Strategic Roadmap: Post-History-Aware Agent Analysis

**Date**: October 18, 2025  
**Status**: 22 experiments complete, architectural limitations identified

---

## Current Situation

We have successfully isolated the core problem preventing agents from matching animal behavior through **22 systematic experiments** across three architectural families:

### The Decoupling Problem (Confirmed)

**What we can model**:

- ✅ **Intra-trial dynamics**: Hybrid DDM+LSTM produces negative chronometric slopes matching or exceeding animal magnitudes (−767 ms/unit vs −645 reference)
- ✅ **Inter-trial heuristics**: Sticky-Q produces qualitatively correct win-stay/lose-shift patterns (0.67/0.38 vs 0.73/0.34 reference)

**What we cannot model**:

- ❌ **Simultaneous intra + inter-trial dynamics**: No agent captures both speed-accuracy tradeoffs AND history-dependent choice biases

### Critical Evidence: History-Aware Agent (Experiment #22)

**Hypothesis**: Explicit history inputs `[prev_choice, prev_reward, prev_stimulus]` would help LSTM bias DDM parameters based on trial history.

**Result**: **NEGATIVE** (run: `history_hybrid_fixed_v1`)

- Win-stay: 0.58 (target 0.73, **20% gap remains**)
- RT: 300ms flat (saturated at response window)
- Psychometric slope: 0.127 (essentially flat)
- Training: 20 epochs, 10 seconds CPU

**Interpretation**: Adding explicit history features to a **static-parameter DDM is insufficient**. The LSTM sets initial DDM parameters once per trial; this cannot create dynamic, history-dependent biases during evidence accumulation. Band-aid confirmed as inadequate.

---

## Root Cause Analysis

### Why Hybrid DDM+LSTM Fails on History

Current architecture:

```text
LSTM(prev_trials) → [drift_rate, bound, non_decision_time] → DDM(stimulus) → choice
         ↑                                                          ↓
         |_____________ one-time parameter setting _________________|
```

**Problem**: DDM parameters are **frozen at trial start**. No mechanism for history to:

1. Bias drift rate dynamically during accumulation
2. Create asymmetric responses to wins vs losses
3. Produce choice stickiness without hand-coded terms

**Evidence**: Across 18+ hybrid runs (including history-aware variant), history metrics consistently hover around 0.5 (chance level). LSTM "coach" does not carry forward reward/choice memory effectively.

### Why Sticky-Q Succeeds on History but Fails on Dynamics

Sticky-Q architecture:

```text
Q(state, action) + α·sticky_prior(prev_choice) → choice
```

**Strengths**: Explicit history term directly biases action selection
**Problem**: Tabular Q-learning with instantaneous action selection cannot learn to **wait for evidence**. No intra-trial integration, no speed-accuracy tradeoff.

---

## Strategic Options (Ranked by Evidence Strength)

### Option 1: Recurrent Drift-Diffusion Model (R-DDM) [RECOMMENDED]

**Core idea**: LSTM influences drift **during** the decision, not just at initialization.

**Proposed architecture**:

```text
At each time-step t:
  drift_t = LSTM(h_t, evidence_t, history) + gain·stimulus
  
  where h_t encodes:
    - Recent trial outcomes (wins/losses)
    - Choice history
    - Reward patterns
```

**Expected outcome**: R-DDM has representational capacity to:

1. Produce negative chronometric slopes (proven in static hybrid)
2. Produce history biases (by modulating drift based on recurrent memory)
3. Unify intra-trial and inter-trial dynamics

**Implementation path**:

1. Start with proven WFPT curriculum from `hybrid_wfpt_curriculum`
2. Replace static DDM parameter predictor with recurrent drift module
3. Add auxiliary loss: `L_history = MSE(win_stay, target) + MSE(lose_shift, target)`
4. Monitor **both** chronometric AND history metrics per epoch

**Estimated effort**: 2-3 days (moderate complexity, builds on existing infrastructure)

**Risk**: Medium. Architecture is more complex but theoretically grounded. May require careful tuning to balance competing loss terms.

---

### Option 2: Rethink DDM's Role (Architectural Pivot)

**Core idea**: Animals may not use DDM-like integration for all decisions.

**Hypothesis**: Perceptual decisions (stimulus → choice) and strategic decisions (history → bias) may use **different neural circuits** that we're forcing into a single DDM framework.

**Possible approaches**:

- **Dual-process model**: Fast perceptual pathway + slow strategic pathway
- **Hybrid policy-DDM**: RL policy handles history, DDM handles stimulus within-trial
- **Hierarchical model**: High-level policy sets "prior belief," DDM integrates stimulus

**Implementation path**: TBD (requires more design work)

**Estimated effort**: 1-2 weeks (substantial redesign)

**Risk**: High. Novel architecture with uncertain payoff. Could be the breakthrough or a dead end.

---

### Option 3: Document and Publish Current Results [DEFENSIVE]

**Core idea**: We have strong negative results showing architectural limitations.

**Deliverables**:

1. Update `README.md` with experiment synthesis
2. Generate comparative dashboards across all 22 runs
3. Write methods paper documenting the decoupling problem
4. Release as benchmark for future work

**Value**:

- Provides clear evidence of what doesn't work
- Establishes reproducible baseline for community
- Clarifies requirements for animal-level behavioral replication

**Estimated effort**: 3-5 days (documentation + visualization)

**Risk**: Low. Valuable regardless of future direction.

---

## Recommended Next Steps

### Immediate (Next 1-2 Days)

1. ✅ **Update FINDINGS.md** with Experiment #22 results (COMPLETE)
2. ✅ **Update registry.json** with history-aware agent documentation (COMPLETE)
3. **Decide on R-DDM vs pivot**: Review architecture designs, discuss implementation plan
4. **Generate cross-experiment dashboard**: Visualize all 22 runs on single page for pattern analysis

### Short-term (Next Week)

If pursuing R-DDM:

1. Design recurrent drift module (sketch math, loss functions, training procedure)
2. Implement minimal R-DDM prototype
3. Run smoke test: 5 epochs, verify gradients flow through recurrent drift
4. Full training: 30 epochs with history loss monitoring

If pivoting:

1. Literature review: Dual-process models, hierarchical RL, neural circuit evidence
2. Sketch 2-3 candidate architectures
3. Implement simplest viable variant
4. Run comparative test vs best hybrid baseline

### Medium-term (Next 2-3 Weeks)

1. **Roadmap alignment**: Ensure architecture generalizes to PRL and DMS tasks (v0.2 scope)
2. **Psychometric calibration**: Address shallow choice slopes (5-10 vs 17.6 reference)
3. **Bias regularization**: Add soft constraints to prevent pathological bias collapse (+6.06 outliers)
4. **Comparative report**: Generate HTML dashboard comparing animal vs all agent families

---

## Decision Criteria

### Success Metrics (Must Achieve Both)

1. **Chronometric slope**: Negative, within 50% of animal reference
2. **History effects**: Win-stay >0.65, lose-shift 0.45-0.55 for IBL; win-stay 0.48-0.52, lose-shift 0.48-0.52 for macaque

### Red Flags (Abort Signals)

- Chronometric slope flattens (history at the cost of dynamics)
- History metrics collapse to 0.5 after 10+ epochs (no learning signal)
- Bias exceeds ±2.0 (pathological repetition/alternation)
- Training takes >30 minutes on CPU (infeasible for iteration)

---

## Open Questions

1. **R-DDM recurrence depth**: Should LSTM hidden state persist across trials or reset?
2. **History loss weighting**: How to balance WFPT (dynamics) vs history supervision?
3. **Macaque extension**: If R-DDM succeeds on IBL, does it generalize to RDM without retraining?
4. **PRL/DMS readiness**: Do R-DDM interfaces support reversal detection and delay periods?

---

## Conclusion

The history-aware agent (Experiment #22) provides definitive evidence that **explicit history inputs to static DDM parameters are insufficient**. This strengthens the case for R-DDM as the necessary architectural change. We are well-positioned to:

1. **Implement R-DDM** with recurrent drift modulation (2-3 days, moderate risk)
2. **Test hypothesis** that dynamic drift can unify intra + inter-trial dynamics
3. **Document results** regardless of outcome (negative results are valuable)

The 22-experiment foundation ensures any next step is evidence-driven rather than speculative. We understand the problem space deeply and have isolated the architectural bottleneck.

---

**Next Action**: Discuss R-DDM implementation plan and decide on immediate priorities (R-DDM prototype vs documentation sprint vs architectural pivot).
