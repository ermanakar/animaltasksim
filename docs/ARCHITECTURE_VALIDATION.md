# Repaired architecture validation — September 11, 2026

This validation separates software integrity, reversal-task runtime interventions,
and exploratory animal prediction. The completed frozen IBL replication is not
rerun or changed. The source animals used here were previously inspected during
development; this is not an untouched prospective confirmation.

## Outcome: the full model did not pass the prediction gate

| Model | Choice NLL (nats/trial) | RT MSE (seconds squared) |
|---|---:|---:|
| Simple one-step history predictor | 0.46450 | 11.24585 |
| Recurrent model without adaptive control | 0.54314 | 11.57947 |
| Full adaptive controller | 0.52579 | 11.62660 |

Lower is better. Values first average training seeds within each animal, then
weight animals equally. Choice scores use all eight animals; RT scores use seven
because SWC_NM_057 has no valid recorded RT. Missing timing data does not remove
that animal's choices. These are separate choice/mean-RT scores, not a joint density.

- Full control improves choice NLL over the reduced recurrent model by 0.01735
  nats/trial, 95% animal-bootstrap interval [0.00237,0.03177]; six of eight improve.
- Full control worsens choice NLL relative to the simple predictor by 0.06129
  nats/trial, 95% interval [0.02633,0.10318]; only one of eight improves.
- RT mean-error advantages are not established against either alternative.
  Full-control improvement intervals are [-1.97074,0.71527] against simple history
  and [-0.15063,0.03985] against no control, in seconds squared.
- Both recorded comparison gates are false. No retraining or selection followed
  these outcomes, and the settings were not changed to obtain a favorable result.

This is evidence that the controller can help its recurrent core under these
settings, while the full fitted system still loses to a simpler predictor. It
neither establishes that the added machinery is generally necessary nor rules
out better optimization or a better-specified response-time model.

The executable run took approximately 329 seconds. Root independently joined all
32,525 stored prediction rows back to source trials and recomputed choice and RT
errors; values matched. The split is disjoint and the recorded plan hash matches.
[Portable result and plan snapshot](results/architecture_validation_v1.json) and
[per-animal/per-seed scores](results/architecture_validation_v1_subject_scores.csv)
retain the numerical evidence.

## Prespecified exploratory prediction gate

The executable plan is `runs/architecture_validation_20260911/prediction/plan.json`,
with a companion SHA-256 digest written before fitting. The runner refuses existing
output directories and checks source stability. Run it with
`python3 -m scripts.validate_architecture_prediction` into a fresh output directory.

- Split by animal identity: 12 training mice and eight different evaluation mice,
  chosen by a deterministic hash ordering, with all their available sessions.
- Compare a logistic choice predictor plus linear mean-RT predictor using stimulus
  and one-trial history against separately trained no-control and full-control
  recurrent models. No model receives block priors or current trial outcomes.
- Fit both recurrent variants for two epochs, with seeds 42 and 123, hidden size 16,
  chronological 64-trial chunks, 32 training diffusion paths and fixed defaults.
- Evaluate hard boundary crossing using 512 paired diffusion paths per trial.
  Inputs include the actual previous animal response and feedback. These are
  conditional next-trial predictions, not autonomous reproduction of the animal.
- Primary measure: equal-animal choice negative log likelihood. Secondary measure:
  equal-animal mean squared RT error, in seconds squared. Lower is better for both.
- Average training seeds within animal, then bootstrap animals 10,000 times. To pass
  this bounded gate, full control must have positive lower 95% improvement limits
  for both measures against both alternatives. No adjustment follows the result.
- Retain all valid RTs in scoring and disclose those outside the model's 50–3000 ms
  response range. Omitted choices are excluded from choice scores and counted.

The baseline receives the same observable fields with two fixed history
interactions and fits on training animals only. It is a simple predictive
alternative, not a proposed biological mechanism. RT MSE tests conditional means,
not the full response-time distribution or a joint choice/RT likelihood.

This is a limited viability comparison: a single split, eight evaluation animals,
two seeds, finite Monte Carlo scoring and two training epochs. Equal epochs do not
establish equal convergence, and intervals do not include training-population or
Monte Carlo uncertainty. A failed gate does not prove that every version of the
architecture must fail; a passed gate would still require independent confirmation.

## Reversal-task runtime check

The lower-cost execution agent verified the existing software tests, lint and
frozen replication hashes, then evaluated four profiles on identical saved weights
with trial-aligned random streams. Each profile produced 1,200 schema-valid trials,
including 12 reversals. All evaluated weights equaled the source checkpoint.

The checkpoint was from the earlier wiring smoke: one epoch on 64 development
trials. Consequently this is a stress test of a lightly trained checkpoint, not a
converged comparison or a PRL animal fit.

| Runtime profile | Optimal-option choices | Reward rate |
|---|---:|---:|
| Full control |55.17%|54.92%|
| Persistence only |55.33%|55.08%|
| Exploration only |62.58%|58.58%|
| No control |52.33%|52.42%|

These numbers do not establish full-control superiority. They concern acute
computational effects in one checkpoint, not retraining effects or neural necessity.
Detailed counts and outcomes are preserved under
`runs/architecture_validation_20260911/pipeline/`.

## Data coverage and software checks

The fixed split contains 14,655 training trials from 20 sessions/12 mice and 6,505
evaluation trials from 9 sessions/eight mice. There are 4,479 evaluation trials
with a valid recorded response time; 348 exceed 3,000 ms. Missing RTs are excluded
from RT scoring, but available choices remain in choice scoring. The evaluation
RT median is 365 ms, 90th percentile 2,156 ms and 99th percentile 15,497 ms. The long tail
is a material mismatch with a model capped at 3,000 ms, not an exclusion criterion.

The expanded repository suite passed 242 tests; the four added tests verify the
subject split, simple baseline learning, omission accounting and equal-animal
aggregation. These are software checks, not evidence that the richer model wins.

## Next decision

Keep the simple predictor as a required benchmark and preserve the full
architecture for diagnosis. Before another confirmatory cohort, use development
data to check optimization/convergence and the response-time observation model,
including missing measurements and the long tail. Compare changes in new labeled
experiments; do not reuse these eight animals as an untouched test set. Defer
additional circuits until the existing model clears a comparable prediction gate.
