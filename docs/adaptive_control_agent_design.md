# Adaptive Control Agent Design

> Status: design proposal, March 2026.
>
> Motivation: March 2026 plastic-history experiments showed that stronger local history plasticity can become numerically active without producing the correct behavioral fingerprint. The next agent family should model not just history bias, but the competing control pressures that make mammals persist, switch, or explore under uncertainty.

## 1. Objective

Design a new agent family for IBL mouse 2AFC that is closer to the likely computational story in mammal brains:

- current sensory evidence influences the decision
- recent reward history influences the decision
- failure does not always imply immediate switching
- some failures should increase persistence
- some failures should increase switching
- some states should trigger exploration because the world is uncertain, stale, or uninformative

The goal is not to hardcode win-stay / lose-shift targets. The goal is to build a control system whose internal pressures naturally produce the fingerprint.

## 2. Why the current history family stalled

The current hybrid agent successfully matches the IBL fingerprint when history is injection-assisted, but repeated March 2026 attempts to replace injected history with learned local plasticity failed in the same way:

- teacher forcing and distillation improved mechanics but not autonomous history learning
- pure plasticity produced stable but weak win-stay
- stronger plasticity made the internal history signal larger without making behavior more animal-like
- asymmetric positive/negative plasticity still looked like a generic recent-history bias

This implies the missing ingredient is not just a stronger history scalar. It is a richer controller that can represent multiple latent causes of behavior after reward or punishment.

## 3. Brain analogy

This proposal is not meant as a literal circuit map. It is a computational analogy motivated by how mouse, monkey, and human brains appear to solve the problem.

### 3.1 Evidence system

Question: what does the stimulus say right now?

Analogy:

- sensory cortex and downstream evidence-accumulation circuits
- in the current repo, this is already approximated by the LSTM + DDM core

### 3.2 Value / outcome system

Question: was the outcome better or worse than expected?

Analogy:

- dopamine-like reward prediction error
- critic-like estimate of expected reward or expected success

### 3.3 Persistence system

Question: should I keep trying this action even though it just failed?

Analogy:

- persistence despite noisy punishment
- retrying on ambiguous trials because the failure may not mean the action is wrong
- a control-like signal associated with effort, conflict, or unresolved uncertainty

This is the part that explains the user's observation: mammals sometimes keep trying after punishment, not because they ignore the punishment, but because they represent uncertainty, delayed usefulness, or the need to probe a strategy further.

### 3.4 Exploration / novelty system

Question: should I try something else because the world might have changed, or because the current strategy is stale?

Analogy:

- novelty seeking
- boredom / habituation
- uncertainty-driven sampling
- information gain

This should not be treated as mere lapse. Lapse is inattentive noise. Exploration is structured sampling.

### 3.5 Arbitration system

Question: which pressure should dominate right now?

Analogy:

- action-selection circuitry that combines evidence, value, persistence, and exploration
- if evidence is strong, sensory evidence should dominate
- if evidence is weak and the previous rewarded action still looks good, stay pressure should dominate
- if evidence is weak and the previous failure is credible, shift pressure should dominate
- if the world is unresolved or stale, exploration pressure should get a vote

## 4. Proposed architecture

Keep the current DDM evidence core and add three new latent drives plus an arbitration head.

### 4.1 High-level structure

```text
stimulus ───────────────► evidence core ───────────────► DDM parameters
previous trial summary ─► value critic ────────────────► value / RPE state
previous trial summary ─► persistence controller ──────► retry pressure
recent action-state stats► exploration controller ─────► novelty pressure
all latent drives ───────► arbitration head ───────────► final action bias
```

### 4.2 Modules

1. Evidence core
   - reuse the existing hybrid LSTM + DDM heads
   - preserve chronometric behavior and current env integration

2. Value critic
   - predicts expected reward or expected correctness for the current context
   - computes a reward-prediction-like signal from observed outcome

3. Persistence controller
   - recurrent or stateful module
   - increases retry pressure when failure occurs under low confidence / weak evidence
   - should not be a simple win/lose lookup

4. Exploration controller
   - tracks novelty, action repetition, and unresolved uncertainty
   - produces a bias toward sampling alternatives when repeated behavior becomes stale or uncertain

5. Arbitration head
   - combines:
     - evidence confidence
     - value estimate / prediction error
     - persistence pressure
     - exploration pressure
   - outputs the final bias that modifies the DDM starting point and/or drift

## 5. Minimal computational formulation

The first version should stay simple enough to train on CPU and fit existing infrastructure.

### 5.1 State variables

Per trial, maintain:

- `value_state_t`: expected reward / success estimate
- `persistence_state_t`: retry pressure after failure under uncertainty
- `exploration_state_t`: novelty / boredom / uncertainty pressure

### 5.2 Suggested updates

Reward prediction error:

$$
\delta_t = r_t - \hat{v}_t
$$

Confidence proxy from the evidence core:

$$
c_t = \mathrm{clip}(|\mathrm{stimulus}|, 0, 1)
$$

Persistence update should grow more after failure when confidence is low:

$$
p_{t+1} = \lambda_p p_t + \alpha_p \cdot \max(-\delta_t, 0) \cdot (1 - c_t)
$$

Exploration update should grow when outcomes are uncertain, stale, or repetitively uninformative:

$$
e_{t+1} = \lambda_e e_t + \alpha_e \cdot u_t + \beta_e \cdot \mathrm{staleness}_t
$$

where `u_t` is a simple uncertainty proxy, initially either:

- low evidence confidence `1 - c_t`, or
- critic prediction error magnitude `|\delta_t|`

Arbitrated bias:

$$
b^{\mathrm{final}}_t = b^{\mathrm{evidence}}_t + g_p p_t + g_e e_t + g_h h_t
$$

where:

- `p_t` pushes retrying the same action under uncertain failure
- `e_t` pushes alternative sampling / exploration
- `h_t` is optional residual short-term history if needed

The critical idea is that a failure does not map directly to a single switch signal. It can increase either persistence or exploration depending on uncertainty and state.

## 6. What changes in the codebase

### 6.1 New modules

Recommended new files:

```text
agents/
  adaptive_control_config.py
  adaptive_control_model.py
  adaptive_control_trainer.py
  adaptive_control_agent.py

scripts/
  train_adaptive_control.py
```

Rationale:

- avoid destabilising the validated hybrid path
- preserve the current best baseline untouched
- make the new agent family independently testable

### 6.2 Reuse from the current hybrid path

Reuse directly:

- differentiable DDM simulator logic
- rollout / env integration patterns
- config persistence
- evaluation pipeline
- curriculum mechanics if helpful

Avoid reusing mechanically:

- the current history-head semantics as the main control story

### 6.3 Frozen contracts

Do not change:

- schema keys
- `.ndjson` ownership by envs
- existing CLI arg names
- existing run layout

New scripts and new internal model classes are allowed. Existing interfaces should remain stable.

## 7. Phase 1 implementation plan

### Phase 1A: Design-compatible scaffold

Goal: build the new agent family without changing scientific outputs yet.

Tasks:

1. create `AdaptiveControlConfig` as a `@dataclass(slots=True)`
2. create `AdaptiveControlModel` with:
   - evidence core wrapper
   - critic head
   - persistence head/state
   - exploration head/state
   - arbitration head
3. create trainer with the same save / rollout shape as the hybrid trainer
4. add smoke tests only

Success criterion:

- end-to-end train/evaluate run completes and logs schema-valid trials

### Phase 1B: First biologically meaningful behavior

Goal: show one behavior the old history family could not represent cleanly.

Recommended target:

- on weak-evidence failures, the agent should show more retry pressure than on strong-evidence failures

This is a better first result than trying to solve the full WS/LS fingerprint immediately.

Success criterion:

- a controlled ablation shows persistence is modulated by evidence confidence after losses

### Phase 1C: Exploration / boredom probe

Goal: show exploration is not just random lapse.

Candidate metric:

- switching probability after repeated predictable outcomes in low-information states

Success criterion:

- exploration rises in stale or uncertain contexts without flattening the psychometric curve

### Phase 1D: Breakthrough probe suite

Goal: make the internal control story falsifiable before spending full validation budgets.

Required probes:

1. weak-evidence failure increases retry pressure
2. high-confidence failure increases switch pressure
3. repeated low-information rewarded choices increase exploration pressure
4. strong current evidence suppresses adaptive control bias

Success criterion:

- each probe passes in isolation, and each pressure is exposed by name in model outputs so later lesion runs can connect internal state to rollout behavior

## 8. Evaluation plan

Do not rely only on win-stay / lose-shift.

Add analysis utilities for:

1. retry-after-failure split by evidence strength
2. switch-after-failure split by evidence strength
3. exploration after repeated predictable runs
4. persistence vs exploration as a function of confidence

The existing evaluation pipeline can remain authoritative for psychometric, chronometric, and standard history metrics. The new metrics can be added as optional analysis outputs first.

## 9. Minimal experiment sequence

1. Build scaffold and smoke test
2. Single-seed prototype with persistence only
3. Single-seed ablation: persistence disabled
4. Add exploration controller
5. Single-seed comparison: persistence-only vs persistence+exploration
6. Only then spend 5-seed budget

## 10. Risks

1. Too many new drives may destabilize training
   - mitigation: persistence-only prototype first

2. Exploration may collapse into random lapse
   - mitigation: keep exploration separate from the fixed lapse mechanism

3. New states may improve new metrics while hurting psych/chrono
   - mitigation: keep the evidence core and drift calibration path intact; evaluate all core fingerprints on every run

4. Overfitting to newly invented metrics
   - mitigation: treat new metrics as probes of the scientific hypothesis, not optimization endpoints at first

## 11. Recommended next implementation

The lowest-risk next coding step is:

1. create a new `AdaptiveControlModel` that wraps the existing DDM evidence core
2. add a critic and a persistence state only
3. route persistence into the DDM bias under low-confidence failure
4. validate with one new metric: retry-after-failure on weak vs strong evidence

That is the smallest change that is both biologically motivated and scientifically testable.
