# Adaptive Control Agent Design

> Status: phase-1 implementation validated, May 2026.
>
> Motivation: March 2026 plastic-history experiments showed that stronger local history plasticity can become numerically active without producing the correct behavioral fingerprint. The next agent family should model not just history bias, but the competing control pressures that make mammals persist, switch, or explore under uncertainty.
>
> Caveat: this is a biologically inspired computational analogy. It is not a claim that the repository models exact brain anatomy.

## Phase-1 result

The phase-1 validation suite now supports a narrow but useful claim:

> The recommended/default `persistence_only` profile preserves calibrated psychometric/chronometric behavior while expressing the validated weak-failure retry mechanism. Full control remains available as a comparison condition, but it includes exploration, which is not independently validated.

Main validation: `runs/adaptive_control_validation_suite_phase1/`

| Condition | Psych slope | Chrono slope | Retry gap | RT ceiling warnings | Degenerate |
|-----------|-------------|--------------|-----------|---------------------|------------|
| true no-control | 27.71 +/- 3.28 | -48.54 +/- 7.05 | 0.057 +/- 0.062 | 0/5 | 0/5 |
| persistence-only | 21.75 +/- 2.69 | -33.47 +/- 4.49 | 0.164 +/- 0.108 | 1/5 | 0/5 |
| full control | 22.26 +/- 1.80 | -33.97 +/- 4.02 | 0.165 +/- 0.045 | 0/5 | 0/5 |

Paired against the clean no-control lesion, full control increased retry gap by `+0.109 +/- 0.086`, positive in 5/5 seeds. Persistence-only recovered almost the same retry-gap mean (`0.164` vs `0.165`) with exploration disabled, so new default/recommended runs should use the `persistence_only` profile unless the purpose is an explicit lesion or full-control comparison.

Gate-lesion validation: `runs/adaptive_control_validation_suite_phase1_gate/`

| Gate condition | Retry lift vs no-control | Positive seeds | Interpretation |
|----------------|--------------------------|----------------|----------------|
| nonlinear gate (`control_uncertainty_power=2.0`) | +0.109 +/- 0.086 | 5/5 | stronger and more reliable |
| linear gate (`control_uncertainty_power=1.0`) | +0.087 +/- 0.130 | 3/5 | still works, but less robust |

This means the nonlinear gate is not strictly necessary. The honest mechanism claim is uncertainty-gated adaptive control, with the sharpened gate improving reliability.

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

Required ablation:

- `control_state_enabled=False` must zero all adaptive-control state outputs, even if persistence and exploration heads are otherwise enabled. This keeps the no-control baseline causally clean: residual behavior can come from the inherited evidence/history core, but not from adaptive-control fast state, retry/switch pressure, exploration pressure, or arbitration.
- Adaptive-control heads must be zero-centered at initialization and routed through a bounded `control_residual`, with training regularization on residual/control pressure magnitude. This keeps persistence and exploration as residual overlays so they cannot erase the evidence core's psychometric and chronometric calibration.
- Adaptive-control state updates use explicit outcome valence: positive reward reinforces the chosen action, while zero/negative reward teaches persistence under uncertainty or switching under confidence. The critic prediction error remains available for value learning and diagnostics, but it must not silence failure teaching when the critic is poorly calibrated.
- Arbitration is gated by current evidence confidence, and training tracks an evidence-preservation penalty on control residuals during high-evidence trials. The adaptive controller should explain ambiguous-trial history effects without rewriting the evidence pathway on easy trials. The expression gate is nonlinear (`control_uncertainty_power`) so control remains available near zero contrast but falls quickly as sensory evidence becomes informative.
- IBL rollout must use the configured DDM response window, not the environment's short default response phase. Otherwise weak-evidence trials saturate at 300 ms and the chronometric warning measures a rollout ceiling rather than agent dynamics.
- Current IBL calibration uses `drift_scale=6.0`, `control_uncertainty_power=2.0`, and `persistence_bias_scale=1.6`. The recommended/default profile is `persistence_only`: it keeps exploration disabled while retaining the validated weak-failure retry mechanism (`retry_gap=0.164` vs `0.057` in the clean no-control lesion). Full control retained calibrated behavior (`psychometric_slope=22.26 +/- 1.80`, `chronometric_slope=-33.97 +/- 4.02`) and produced the most consistent paired retry lift (`delta_retry_gap=+0.109 +/- 0.086`, positive in 5/5 seeds), but it should be labeled as a comparison condition because exploration is not validated. No full-control runs were degenerate or RT-ceiling flagged.
- Gate-lesion validation showed that a linear uncertainty gate still produces some adaptive retry behavior (`delta_retry_gap=+0.087 +/- 0.130`) but less reliably (3/5 positive seeds). Do not claim the nonlinear exponent is necessary; claim it improves robustness.

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

Use `scripts/adaptive_control_validation_suite.py` for the matched validation run. It trains the clean no-control lesion, exploration-only condition, persistence-only recommended condition, and full-control comparison condition across seeds, then writes per-run summaries, aggregate summaries, and paired deltas against the no-control lesion.

The suite now reports a first-class stale-exploration probe:

- `switch_after_streak_weak`: switch rate on weak-evidence trials after a rewarded same-action streak.
- `switch_after_fresh_weak`: switch rate on weak-evidence trials without a stale rewarded streak.
- `stale_switch_lift_weak`: `switch_after_streak_weak - switch_after_fresh_weak`; this is the primary exploration readout.
- `exploration_gap`: older secondary readout comparing weak-streak switching against strong-streak switching.

The `exploration_only` lesion disables persistence but leaves the exploration controller and control state enabled. This makes the exploration claim cleaner: the expected result is not just “full control changed behavior,” but “stale-state exploration specifically increases weak-evidence switching above the fresh weak-evidence baseline.”

The first 5-seed exploration run did **not** support that expected result. In `runs/adaptive_control_validation_suite_phase1_exploration/`, `stale_switch_lift_weak` was negative in every condition and became more negative relative to no-control:

- exploration-only minus no-control: `-0.087`, positive in `0/5` seeds
- persistence-only minus no-control: `-0.086`, positive in `0/5` seeds
- full-control minus no-control: `-0.079`, positive in `0/5` seeds

So the supported phase-1 claim remains persistence/adaptive retry after weak-evidence failure. Rewarded-streak exploration is not validated and is disabled in the recommended/default training profile. The next exploration work should either define a better probe around unrewarded or volatile streaks, or defer the exploration claim to PRL/DMS where environmental change is task-relevant.

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

## 11. Recommended next work

The scaffold and phase-1 validation are complete. The next scientific steps are:

1. isolate the exploration component with an exploration-disabled lesion and stronger stale-state probes
2. test whether the adaptive-control idea transfers to PRL/DMS-style tasks
3. keep the claim narrow: biologically inspired control mechanism, not exact anatomy
4. avoid spending more budget on the nonlinear gate unless a task-transfer result makes it decisive
