# Probabilistic Reversal Learning Transfer Design

## Why This Task Exists

IBL 2AFC asks whether an agent handles uncertain sensory evidence in an
animal-like way. Probabilistic Reversal Learning (PRL) asks a different
question: can the same controller notice that the world changed and adapt from
outcomes alone?

Think of two identical vending buttons. One pays out 80% of the time and the
other pays out 20% of the time. After a while, the payouts silently swap. The
buttons must still look identical. If one button changes color, the task stops
measuring inference and starts measuring whether the agent noticed the hint.

## Faithful Environment Rules

`envs/prl_reversal.py` implements the PRL environment:

- both visible options are neutral: `stimulus.contrast = 0.0`
- one hidden contingency block favors left and the next favors right
- adjacent blocks, including the wraparound, must reverse the favored option
- reward remains probabilistic, so one unlucky outcome is not perfect evidence
- reversals and contingencies are logged for offline analysis only
- the policy receives previous action, reward, and correctness through the
  existing adaptive-control history path; it never receives an oracle reversal
  signal

The environment owns `.ndjson` logging and validates every trial through the
shared schema.

## Transfer Protocol

The PRL slice is a zero-shot transfer probe for the adaptive-control family:

1. Train the shared evidence core against `data/ibl/reference.ndjson`.
2. Roll the same controller into PRL with neutral sensory evidence.
3. Ask whether reward history and adaptive control produce recovery after
   hidden payout reversals.
4. Compare matched lesions rather than reading one seed as a result.

The IBL reference core is reused because there is not yet a PRL animal reference
dataset in this repository. That makes this a transfer test, not a PRL animal
parity claim.

## PRL Fingerprint

`eval/metrics.py` reports:

| Metric | Meaning |
|--------|---------|
| `optimal_choice_rate` | Fraction of committed choices aimed at the richer hidden option |
| `reward_rate` | Fraction of committed trials that paid out |
| `early_optimal_choice_rate` | Optimal choices on trials 1-5 after reversal |
| `late_optimal_choice_rate` | Optimal choices on trials 6-10 after reversal |
| `adaptation_lift` | Late minus early optimal-choice rate |
| `end_block_optimal_choice_rate` | Optimal choices on the final 20 trials before the next reversal |
| `block_learning_lift` | End-block minus early optimal-choice rate |
| `stay_after_rewarded` | Tendency to repeat a choice after payout |
| `switch_after_unrewarded` | Tendency to switch after no payout |

`reversal_probe_ok` and `block_learning_probe_ok` mean the run crossed at least
one reversal and produced finite readouts. They do not mean the scientific
hypothesis passed.

## Practical Commands

Small lifecycle smoke:

```bash
python scripts/train_adaptive_control.py \
  --task prl \
  --control-profile persistence_only \
  --output-dir runs/prl_adaptive_control_smoke \
  --seed 42 --episodes 1 --trials-per-episode 170 \
  --epochs 0 --hidden-size 16 \
  --max-sessions 1 --max-trials-per-session 10
python scripts/evaluate_agent.py --run runs/prl_adaptive_control_smoke
python scripts/make_report.py --run runs/prl_adaptive_control_smoke
```

Matched transfer experiment:

```bash
python scripts/prl_transfer_validation_suite.py \
  --run-root runs/prl_transfer_validation_suite \
  --seeds 42 123 456 789 2026
```

The suite compares:

1. `true_no_control`
2. `exploration_only`
3. `persistence_only`
4. `full_control_default`
5. `full_control_persist_half`

## Matched Result

The May 30, 2026 suite completed with 25 usable runs: five paired seeds for each
condition, 1,600 trials and 16 reversals per run, and 0/25 degenerate runs.

| Condition | Overall optimal choice | End-block optimal choice | Block-learning lift |
|-----------|------------------------|--------------------------|---------------------|
| `true_no_control` | 0.504 | 0.513 | +0.053 |
| `exploration_only` | 0.579 | 0.683 | +0.360 |
| `persistence_only` | 0.469 | 0.492 | +0.019 |
| `full_control_default` | 0.507 | 0.466 | -0.044 |
| `full_control_persist_half` | 0.510 | 0.543 | +0.066 |

`exploration_only` beats no-control on end-block optimal choice by `+0.169` and
on block-learning lift by `+0.307`, both positive in 5/5 paired seeds. It beats
`persistence_only` on block-learning lift by `+0.341`, also positive in 5/5.

## Arbitration Scale Sweep Result

The follow-up interaction sweep completed on May 30, 2026 under
`runs/prl_adaptive_control_interaction_sweep_v1/`: 10 conditions, five paired
seeds, 80,000 schema-valid trials, 100% commit rate, and 0/50 degenerate runs.

| Full-control condition | Persistence scale | Exploration scale | End-block optimal choice | Block-learning lift |
|------------------------|-------------------|-------------------|--------------------------|---------------------|
| `full_control_default` | 1.6 | 0.8 | 0.466 | -0.044 |
| `full_control_persist_half` | 0.8 | 0.8 | 0.543 | +0.066 |
| `full_control_persist_quarter` | 0.4 | 0.8 | 0.459 | -0.094 |
| `full_control_explore_strong` | 1.6 | 1.2 | 0.444 | -0.126 |
| `full_control_explore_double` | 1.6 | 1.6 | 0.534 | +0.057 |
| `full_control_balanced` | 0.8 | 1.2 | 0.445 | -0.080 |
| `full_control_explore_dominant` | 0.4 | 1.6 | 0.454 | -0.016 |

Every full-control variant loses block-learning lift versus `exploration_only`
in 5/5 paired seeds. Loading the saved checkpoints also showed that trained
scales stayed close to their configured starts, so the grid did test distinct
scale settings. Global scale tuning is not enough to solve the interaction.

## Claim Boundary

This supports an exploration-specific hidden-contingency transfer phenotype
when the exploration controller is isolated. It does not validate PRL animal
parity, because this repository has no PRL animal reference dataset. It also
does not validate the combined full-control controller: all tested full-control
variants suppress most of the exploration-only recovery effect.

DMS is one step behind PRL: its environment and schema path exist, and its
memory-specific fingerprint is defined in `docs/dms_memory_fingerprint_design.md`.
Adaptive-control rollout remains deliberately unwired until the evaluator and a
memoryless baseline exist.

## Follow-up Experiment

The checkpoint-reroll sidecar diagnostic used for the follow-up was:

```bash
python scripts/prl_arbitration_diagnostic.py \
  --source-root runs/prl_adaptive_control_interaction_sweep_v1 \
  --output-root runs/prl_arbitration_diagnostic_v1
```

The default command rerolls `exploration_only`, `full_control_persist_half`,
and `full_control_explore_double` across the same five seeds. It records
`control_bias`, persistence pressure, exploration pressure, arbitration
adjustment, and final bounded residual across each hidden-contingency block.
These values live in a separate `control_diagnostics.ndjson` sidecar;
`trials.ndjson` remains frozen.

The decomposition localized the deficit to `uncertain_retry`: neutral PRL
options pin the old stimulus-derived uncertainty signal at 1.0, so retry fires
after every failure. The follow-up change-evidence recurrence accumulates
recent failures and uses them to close retry while opening switch behavior.
Safety-gated calibration selected λ=0.9 as the validated opt-in cross-task profile:
with `uncertain_retry` still enabled, full control reaches PRL block-learning
lift `+0.469` and optimal choice `0.706`. After the June 1 prior-trial
retry-metric correction, its IBL retry gap is `0.158` versus historical
flag-off `0.175`. The feature remains default off.
