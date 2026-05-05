# Adaptive Control Exploration Probe Design

> Status: metric-first probe scaffold, May 2026.

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

A second probe asks whether the agent switches more after recent mixed outcomes. This is still only an IBL-compatible proxy for volatility; PRL/DMS will be a better transfer test once those tasks land.

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

## Next Experiment Shape

Use the existing four-condition matched validation suite:

1. `true_no_control`
2. `persistence_only`
3. `exploration_only`
4. `full_control`

Read the new lifts alongside the existing retry and stale-switch metrics. Do not tune agent parameters against a single-seed positive result; treat the first pass as a falsification screen.
