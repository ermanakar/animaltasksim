# Adaptive Control Exploration Probe Design

> Status: first 5-seed falsification screen complete, May 2026.

## Goal

The previous rewarded-streak exploration probe was a useful negative result: in stable IBL 2AFC, repeated reward may correctly invite exploitation, not exploration. The next probe should therefore ask a sharper question:

> Does the exploration controller increase structured switching after repeated weak failures or locally volatile outcomes?

This is deliberately a metric-first design. The evaluator should define the readout before any agent tuning or sweep spends a validation budget.

## Probe 1: Unrewarded Weak-Failure Streaks

Repeated unrewarded choices under weak evidence are the cleanest place to look for exploration. A failure on an ambiguous trial can mean "try again" or "sample another option"; the persistence claim already covers the retry side. The exploration claim needs to show that switching rises when failures accumulate.

Primary readouts in `exploration_probe`:

- `switch_after_unrewarded_streak_weak`: switch rate on weak-evidence trials after at least two consecutive weak-evidence failures.
- `switch_after_unrewarded_fresh_weak`: matched weak-evidence switch rate without that failure streak.
- `unrewarded_switch_lift_weak`: streak switch rate minus fresh switch rate.
- `unrewarded_streak_weak_count` and `unrewarded_fresh_weak_count`: event counts; low counts make the lift descriptive only. A two-failure threshold is used because three-failure streaks were effectively absent in the one-seed count screen.

Expected exploratory signature:

- `unrewarded_switch_lift_weak > 0`
- paired deltas versus no-control and persistence-only are positive
- the effect is strongest in `exploration_only` or `full_control`, not in `persistence_only`

## Probe 2: Local Outcome Volatility

A second probe asks whether the agent switches more after recent mixed outcomes.
This is still only an IBL-compatible proxy for volatility. PRL has now landed as
the better hidden-contingency transfer test; DMS remains a future memory probe.

Primary readouts in `exploration_probe`:

- `switch_after_volatile_weak`: switch rate on weak-evidence trials after a recent window containing both successes and failures.
- `switch_after_stable_weak`: matched weak-evidence switch rate after locally stable outcomes.
- `volatile_switch_lift_weak`: volatile switch rate minus stable switch rate.
- `volatile_weak_count` and `stable_weak_count`: event counts.

Expected exploratory signature:

- `volatile_switch_lift_weak > 0`
- paired deltas versus no-control and persistence-only are positive
- the effect does not flatten psychometric or chronometric behavior

## Falsification Rules

This probe should be allowed to fail. It does not validate exploration if:

- lifts are zero or negative across paired seeds
- positive means come from one seed while positive-seed counts are weak
- event counts are too low to interpret
- full-control passes but `exploration_only` does not, unless a documented interaction explains why
- psychometric/chronometric quality collapses

## May 6, 2026 Screen Result

Run: `runs/adaptive_control_exploration_probe_5seed/`

This was a lightweight matched screen across `true_no_control`, `persistence_only`, `exploration_only`, and `full_control` (5 seeds, 3 episodes, 1 epoch). It was designed to answer whether the new probes have enough event support and whether either one separates exploration from persistence before spending a larger validation budget.

Quality checks were clean: all four conditions had 0/5 degenerate runs and 0/5 RT-ceiling warnings.

Paired deltas versus no-control:

| Comparison | Delta retry gap | Retry positive seeds | Delta unrewarded-switch lift | Unrewarded positive seeds | Delta volatile-switch lift | Volatile positive seeds |
|------------|-----------------|----------------------|------------------------------|---------------------------|----------------------------|-------------------------|
| exploration-only - no-control | +0.038 | 3/5 | +0.037 | 4/5 | +0.054 | 3/5 |
| persistence-only - no-control | +0.082 | 5/5 | -0.143 | 2/5 | +0.040 | 3/5 |
| full-control - no-control | +0.082 | 5/5 | -0.008 | 4/5 | +0.071 | 4/5 |

Paired deltas versus persistence-only:

| Comparison | Delta retry gap | Retry positive seeds | Delta unrewarded-switch lift | Unrewarded positive seeds | Delta volatile-switch lift | Volatile positive seeds |
|------------|-----------------|----------------------|------------------------------|---------------------------|----------------------------|-------------------------|
| exploration-only - persistence-only | -0.044 | 2/5 | +0.180 | 4/5 | +0.014 | 3/5 |
| full-control - persistence-only | -0.000 | 2/5 | +0.135 | 4/5 | +0.032 | 4/5 |

Interpretation:

- The validated retry effect remains a persistence/control-state result. Persistence-only and full-control both increased retry gap by about `+0.082` versus no-control, positive in 5/5 seeds.
- The unrewarded-failure lift is too thin to carry a claim. Streak counts averaged about 9-12 events per run, so the positive exploration-vs-persistence deltas should be treated as descriptive.
- The local-volatility lift is the better candidate readout because it has hundreds of events per run and is positive for full-control versus no-control (`+0.071`, 4/5 seeds). It is not yet exploration-specific because persistence-only also moves positively versus no-control (`+0.040`, 3/5 seeds).
- Exploration remains experimental/unvalidated. A future claim needs positive paired deltas versus both no-control and persistence-only, with enough event counts and no psychometric/chronometric degradation.

## Next Experiment Shape

Use the existing four-condition matched validation suite:

1. `true_no_control`
2. `persistence_only`
3. `exploration_only`
4. `full_control`

Read the new lifts alongside the existing retry and stale-switch metrics. Do not tune agent parameters against a single-seed positive result. The `block_switch_probe` uses uncued IBL block reversals as a bridge to PRL: exploration should help the agent move toward a new hidden prior without merely increasing random switching.

## Block-Switch Follow-Up Result

Run: `runs/adaptive_control_block_switch_focus_v1/`

The block-switch-focused suite increased rollout support to 30 biased block reversals per run. Exploration-only improved block-switch adaptation versus no-control (`+0.103`, positive in 5/5 seeds) and versus persistence-only (`+0.037`, positive in 4/5 seeds). This is the first strong lead for an exploration-specific behavioral role.

Caveat: full-control did not inherit that advantage over persistence-only (`-0.015`, positive in 0/5 seeds). The next experiment should therefore isolate the arbitration/interaction between persistence and exploration, not simply declare full-control validated.

## May 7 interaction sweep

Run: `runs/adaptive_control_interaction_sweep_v1/`

The arbitration sweep tested whether full-control can preserve the block-switch lead by changing persistence/exploration scales. The cleanest rescue was `full_control_persist_half` (`persistence_bias_scale=0.8`, `exploration_bias_scale=0.8`): block-switch lift was `+0.136`, equal to exploration-only and `+0.037` above persistence-only in 5/5 paired seeds. The default full-control setting remained below persistence-only (`-0.015`, 0/5 seeds), so the interaction failure was scale-dependent rather than a total failure of the full-control path.

This still does not validate exploration as the main project claim. `full_control_persist_half` reduced retry gap relative to persistence-only (`-0.025`, 0/5 positive seeds), so it should be treated as a promising full-control comparison/transfer candidate. The recommended/default claim profile remains `persistence_only`.

### PRL transfer follow-up

The matched hidden-contingency PRL suite completed on May 30, 2026. It gave the
exploration lead a stronger out-of-sample test:

| Condition | End-block optimal choice | Block-learning lift |
|-----------|--------------------------|---------------------|
| no-control | 0.513 | +0.053 |
| exploration-only | 0.683 | +0.360 |
| persistence-only | 0.492 | +0.019 |
| full-control default | 0.466 | -0.044 |
| full-control persist-half | 0.543 | +0.066 |

Exploration-only beats no-control by `+0.307` block-learning lift and
persistence-only by `+0.341`, positive in 5/5 paired seeds for both
comparisons. The useful negative result is that the combined controllers still
mask most of this effect. PRL therefore validates the isolated exploration
mechanism lead while reopening the arbitration problem.

### PRL arbitration scale sweep

Run: `runs/prl_adaptive_control_interaction_sweep_v1/`

The PRL interaction sweep tested seven combined full-control scale variants
across five paired seeds. All 50 runs were usable and all 80,000 trial records
passed schema validation. No combined profile preserved the exploration-only
learning curve. The best combined block-learning lifts were only `+0.066` for
`full_control_persist_half` and `+0.057` for `full_control_explore_double`,
versus `+0.360` for `exploration_only`.

The next experiment did not spend more budget on global scale tuning. The
sidecar reversal-window diagnostic showed that PRL's neutral options pin the
old stimulus-derived uncertainty signal at 1.0, so `uncertain_retry` fires
after every failure. The state-dependent change-evidence recurrence now
accumulates recent failures and closes that retry gate while opening the switch
gate:

```bash
python scripts/prl_arbitration_diagnostic.py \
  --source-root runs/prl_adaptive_control_interaction_sweep_v1 \
  --output-root runs/prl_arbitration_diagnostic_v1
```

Safety-gated calibration rejected λ=0.7 as too eager and selected λ=0.9 as the
leading opt-in combined profile. With `uncertain_retry` still enabled, λ=0.9
full control reaches PRL block-learning lift `+0.469` and optimal choice
`0.706`; its IBL retry gap is `0.115` versus the historical flag-off `0.165`.
The feature remains default off while that IBL tradeoff is studied.
