# Confirmation on untouched animals — protocol, November 26, 2026

Milestone 5 of the [research plan](RESEARCH_PLAN.md). The
[within-animal comparison](WITHIN_ANIMAL_COMPARISON.md) and the
[state predictions](STATE_PREDICTIONS.md) were exploratory: they used animals
the project had already examined. This protocol tests the same claims, with
the same code and settings, on animals whose trials the project has never
downloaded.

**Status: confirmatory, internally frozen.** It is not an external
preregistration. This document is a draft until the `plan` stage records its
hash; after that it is not edited, and results go in a separate report.

This document was written before any trial of a confirmation animal was
requested.

## What was already seen

- Everything in the reports of milestones 1 to 4, all on development animals.
- **Session metadata of the confirmation animals.** The sampling frame saved
  for the September replication (`runs/ibl_replication_plan/sampling_frame.json`,
  5,217 sessions, retrieved September 6, 2026) lists session id, animal, lab,
  start time and task protocol for every public `biasedChoiceWorld` session.
  It holds no trial, performance or quality field. It was used here to count
  animals and sessions and for nothing else.
- The confirmation script was exercised on development animals only: every
  stage after download on six of them in a scratch directory, and the summary
  on the stored development scores, where all five claims read "confirmed" at
  99%, as expected for the animals the claims came from.
- No trial array, choice, response time or quality record of any confirmation
  animal has been requested by this project. As in the replication protocol,
  earlier unrecorded access cannot be excluded.

## Cohort

Fixed by rule from the saved frame, with no new listing.

- **Animals:** every animal in the frame that is neither one of the 90 animals
  of the development references nor one of the 60 animals reserved for the
  September replication, and that has at least six sessions with an exact
  versioned standard `_iblrig_tasks_biasedChoiceWorld` protocol.
  This gives **43 animals and 855 sessions** from eight labs (one lab
  contributes 23).
- **Sessions:** exactly those 855. Optogenetic, RPE and every other named
  variant are excluded, as before. Sessions recorded after the frame date are
  not added.
- **All 43 are used.** The pool is too small to hold part of it back, so after
  this run the project has no untouched animals left for this question.
- The runner fails closed if a development or replication animal appears.

### Eligibility after download

Structural rules only, identical to the development analysis, never based on
performance, history or state measures:

- conversion by the [frozen conversion rules](DEEP_CONVERSION_PROTOCOL.md);
- a session counts if it has at least 150 trials;
- an animal is eligible if it has at least six such sessions.

Every planned session is attempted. Download failures are retried for the same
session ids. Sessions without retrievable trials are recorded and not replaced.
No animal is replaced.

## Sample size and stopping

- There is one analysis, run once, on all eligible animals. No interim look,
  no extension, no second cohort under this protocol.
- **Minimum:** 30 eligible animals. With fewer, the result is "insufficient
  sample": numbers are reported and no claim is confirmed or refuted.
- `run` prints no score. `summarize` computes everything once and refuses to
  overwrite its result.

Planning estimates, from resampling the per-animal values of the development
animals (probability that a claim passes its rule below):

| Claim | 30 animals, same size | 30 animals, half size | 40 animals, half size |
|---|---|---|---|
| C1, C2 | > 0.99 | > 0.99 | > 0.99 |
| P1 | 0.99 | 0.83 | 0.92 |
| P2 | 0.98 | 0.91 | 0.97 |
| P3 | 0.74 | 0.56 | 0.67 |

P3 has modest power and will also lose animals with fewer than ten usable
omissions. A failure of P3 alone would not distinguish "absent" from "too few
events".

## Methods

Unchanged from development. No setting is tuned on confirmation data, and the
analysis modules are used at the versions whose hashes the plan records.

- **Models and scoring:** as in the
  [comparison protocol](WITHIN_ANIMAL_COMPARISON_PROTOCOL.md): six models,
  ridge penalty 1, 41 drift states, three switching states, iteration limit
  5,000, fitted per animal on even-numbered sessions and scored on
  odd-numbered ones and the reverse; mean negative log likelihood per
  committed trial.
- **State estimate and measures:** as in the
  [state predictions protocol](STATE_PREDICTIONS_PROTOCOL.md), including the
  regression controls, the omission measure and the minimum counts.

## Claims and decision rules

Five claims are tested. To hold the chance of any false confirmation near 5%
across the five, every interval here is a **99%** bootstrap interval over
animals (10,000 resamples). The thresholds are the ones used in development.
A 99% interval can only make detection harder than the 95% rule the recovery
study validated.

**Primary** (held-out score, δ = 0.001 nats per trial):

| | Claim | Contrasts | Confirmed if |
|---|---|---|---|
| **C1** | A slow latent state improves prediction beyond trial history | `history` − `drift_history`; `history` − `switching_history` | the lower bound exceeds δ for at least one |
| **C2** | Trial history improves prediction beyond the state | `drift` − `drift_history`; `switching` − `switching_history` | the lower bound exceeds δ for both |

**Secondary** (state predictions):

| | Claim | Measure | Confirmed if |
|---|---|---|---|
| **P1** | Choices towards the side the state favours are faster | choice × b̂ on log response time | the upper bound is below −0.02 |
| **P2** | A stronger lean goes with slower responses | \|b̂\| on log response time | the lower bound exceeds 0.02 |
| **P3** | The state leans more strongly before omitted trials | adjusted \|b̂\|, omitted − committed | the lower bound exceeds 0.02 |

For each claim the reading is one of:

- **confirmed:** the rule above is met;
- **contradicted:** the 99% interval lies wholly on the other side of zero
  (for C1 and C2, for both contrasts);
- **not confirmed:** anything else. The estimate and interval are reported.
  This is not evidence that the effect is absent.

**Reported without a claim:**

- The form of the state (`drift_history` − `switching_history`). Development
  left it unresolved, and 43 animals cannot resolve a difference of that size.
- The block-change lag D1, with its 95% interval and the development estimate
  (−0.024) beside it.
- Everything else the two development reports listed: gains over `fixed`, the
  share of the history gain remaining after state, counts of animals with a
  gain, fitted drift parameters, means by lab.

## Conditions on the run

Checked at the summary. If a condition fails, the affected claims are reported
as numbers and **no reading is drawn** from them.

1. At least 30 eligible animals (all claims).
2. At most 2% of selected fits report non-convergence (C1, C2).
3. Refitting both drift models with 81 states changes held-out scores by at
   most 2 × 10⁻⁴ nats per trial on average (C1, C2).
4. The synthetic method check of the state predictions, repeated on the
   confirmation schedules, passes its gate before any real response time is
   read (P1, P2, P3).

Condition 1 applies to every claim. The method check uses the 95% readings of
its own protocol, unchanged.

## Planned sensitivity analyses

They show whether a reading depends on a choice made here. They do not replace
the primary reading. A claim whose reading changes is reported as not robust.

- **Without the largest lab.** One lab contributes about half of the animals.
- **Chronological split** for C1 and C2.
- **95% intervals**, for direct comparison with the development reports.

## What each outcome will be taken to mean

- **C1 confirmed:** within mice not used to develop the analysis, a slowly
  changing side bias predicts held-out choices better than trial history
  alone. It is a statement about prediction, not about a named internal state.
- **C1 not confirmed or contradicted:** the direction adopted on November 23
  does not generalize. It is reported as such and the agent work of milestone
  6 does not start from it.
- **C2 confirmed:** trial history is not only a by-product of the slow state.
- **P1 to P3:** as in the state predictions protocol. Confirmation still does
  not separate "a hidden variable" from "the animal is slower and misses more
  during one-sided runs"; this protocol does not attempt that.
- Development and confirmation estimates are shown side by side. A confirmed
  claim with a clearly smaller effect is reported as confirmed and smaller.

## Limits

- Internal freeze, one analyst, no external registration.
- The same public dataset, task, rigs and labs as development. This is not
  an independent experiment, and it says nothing about other tasks or species.
- Session metadata of these animals has been held since September.
- About half of the animals come from one lab, as in development.
- The animals have fewer sessions than the development animals (median 16
  against 26). The recovery study validated the rules on the development
  schedules; it was not repeated for these.
- The claims and thresholds were shaped by the development results. That is
  what a confirmation is for, but it means the claims are the ones development
  favoured.
- Model fits and associations do not establish mechanisms in animals.

## Stages

One script, written and tested without access to confirmation trials:

```bash
python -m scripts.confirm_within_animal --stage cohort     # from the saved frame only
python -m scripts.confirm_within_animal --stage plan       # hashes this file, the code and the cohort
python -m scripts.confirm_within_animal --stage download   # first access to confirmation trials
python -m scripts.confirm_within_animal --stage convert
python -m scripts.confirm_within_animal --stage check      # synthetic method check
python -m scripts.confirm_within_animal --stage run
python -m scripts.confirm_within_animal --stage summarize
```

Outputs go to a new local directory, `runs/confirmation_v1`. Portable copies of
the result go under `docs/results/`.
