# Within-animal recovery protocol — November 24, 2026

Milestone 2 of the [research plan](RESEARCH_PLAN.md). Before any model is
fitted to real choices in the deep reference, this study asks whether the
planned comparison can tell fixed behaviour, trial history, slow drift and
discrete states apart **at the real data depth**. It uses the animals' real task
schedules and entirely synthetic choices. No real choice or outcome is used.

This document was written before the recovery simulations were run. Model code
existed and had been timed on random inputs; nothing had been simulated from
the generators below.

After the protocol was written and before the study ran, each generator was run
once on an invented schedule to check that the code executes, and unit tests
were written. No setting, threshold or generator value was changed afterwards.

## Why the earlier recovery study is not reused

The [September study](GENERATIVE_COMPARISON.md) fitted models across animals
with about 1.4 sessions each, used a drift model with fixed parameters, and
failed its gate. The deep reference changes the design: each model is now
fitted **within one animal**. Animals therefore share no fitted parameter, which
this protocol relies on.

## Data used

Schedules only: contrast, block prior and omission positions of every converted
biased-family session with at least 150 trials, for animals with at least six
such sessions (71 animals, median 26 sessions and about 19,000 trials each).
The length rule is structural. No performance-based filter is applied, because
low performance may itself be a state. Training and ephys sessions are not used.

## Models

Six models, each fitted separately per animal by penalised likelihood:

| Model | Description |
|---|---|
| `fixed` | One weight per signed contrast and one block term |
| `history` | `fixed` plus rewarded and unrewarded choices at lags 1–5 and two weak-evidence terms |
| `drift` | `fixed` plus a latent side bias following an AR(1) process whose persistence and size are **fitted** |
| `drift_history` | `drift` plus the history terms |
| `switching` | Three states, each with its own task weights, and a fitted transition matrix |
| `switching_history` | Three states, each with its own task and history weights |

Fixed settings: ridge penalty 1 on every weight; drift discretised on 41 states
spanning ±3 stationary standard deviations, persistence bounded to
[0.90, 0.9995]; switching transitions with Dirichlet pseudo-count 0.1 and two
deterministic starts, the lower training objective kept. Latent states restart
at each session. An omitted trial carries no choice information, clears the
history terms, and lets the latent state evolve. These settings are not tuned
on held-out scores.

## Scoring

Sessions are numbered in time order within each animal. Models are fitted on
even-numbered sessions and scored on odd-numbered ones, then the reverse, so
every committed trial is scored once by a model that did not see its session.
A prediction uses only earlier trials of the same held-out session. The score is
mean negative log likelihood per committed trial, per animal.

## Endpoints

These are the endpoints intended for milestone 3. Each is an equal-animal mean
of a per-animal difference in held-out score, with a 95% bootstrap interval over
animals. The smallest difference of interest is **δ = 0.001 nats per trial**,
about the size of the replicated evidence effect and well below every main
effect seen in development.

| Endpoint | Contrasts | Detection rule |
|---|---|---|
| **State** beyond trial history | `history` − `drift_history`; `history` − `switching_history` | lower bound above δ for at least one |
| **History** beyond state | `drift` − `drift_history`; `switching` − `switching_history` | lower bound above δ for both |
| **Form** of the state | `drift_history` − `switching_history` | switching if the lower bound is above δ; drift if the upper bound is below −δ; otherwise unresolved |

## Generators

Choices are simulated per animal on its own schedule. Every animal draws a
sensitivity multiplier (log-normal, σ 0.2) on a sensory term of 4 × signed
square-root contrast, a constant side bias (normal, SD 0.4) and a block term of
±0.4. Zero-contrast trials draw their rewarded side from the block prior.
Strengths are set from numbers already recorded in development, not fitted.

- **History, strength 1.** Lag-1 weights +0.30 after reward and −0.25 after
  error, decaying by 0.6 per lag to lag 5; weak-evidence terms +0.15 after
  reward and −0.30 after error. Each animal scales all of these by a log-normal
  factor (σ 0.4). Values follow the net weights in the
  [first look](DEEP_EXPLORATION.md) and the coefficients in the
  [outcome split](OUTCOME_SPLIT.md).
- **Drift.** AR(1) side bias with persistence 0.995 per trial and stationary SD
  0.6 scaled per animal (log-normal, σ 0.3), restarted from its stationary
  distribution each session.
- **Switching.** An engaged state and two biased states with sensory weight
  × 0.35 and bias ∓1.5. Per trial, engaged moves to each biased state with
  probability 0.01; a biased state returns to engaged with 0.03 and crosses to
  the other with 0.005.

Eight cells: `fixed`, `history`, `drift`, `switching`, `drift_history`,
`switching_history`, and the last two again with history at strength 0.5.

## Replication and the cohort-level estimate

Each cell is simulated with seeds 0, 1 and 2, giving 213 simulated animals per
cell. Because every model is fitted within one animal, simulated animals are
independent. The probability that a 71-animal study triggers a detection rule
is therefore estimated by drawing 2,000 cohorts of 71 with replacement from the
213 simulated animals of a cell and applying the rule to each (500 bootstrap
resamples per cohort).

## Gate

All of the following, with no threshold changed afterwards:

1. **State, false detection ≤ 0.05** in `fixed` and `history`.
2. **State, power ≥ 0.90** in `drift`, `switching`, `drift_history` and
   `switching_history` at strength 1.
3. **History, false detection ≤ 0.05** in `fixed`, `drift` and `switching`.
4. **History, power ≥ 0.90** in `history`, `drift_history` and
   `switching_history` at strength 1.
5. **Form:** the true form is named with probability ≥ 0.90 and the wrong form
   with probability ≤ 0.05, in the four cells that contain a state.
6. **Convergence:** at most 2% of selected fits in any cell report
   non-convergence.
7. **Numerical resolution:** on the seed-0 `drift` cell, refitting both drift
   models with 81 states changes no animal's held-out score by more than
   2 × 10⁻⁴ nats per trial on average across animals.

The two half-strength cells are reported and not gated. They show where
detection of history begins to fail.

Also reported, not gated: the full matrix of generator against best-scoring
model per animal, fitted drift parameters against their true values, and every
mean gain with its interval.

## What a result will mean

- **Pass:** the endpoints can separate these alternatives at these strengths on
  these schedules. Milestone 3 may be frozen with the same models and endpoints.
- **Fail:** the failing criterion is the finding. Milestone 3 does not proceed
  on that endpoint until the design is changed in a new dated protocol.

## Limits

- Generator strengths are anchored to development numbers but not fitted. Real
  behaviour may differ in ways no cell represents, including several slow
  processes at once, lapses, or states that persist across sessions.
- The switching generator and the switching model both have three states. A
  mismatch in the number of states is not tested.
- The cohort-level estimate resamples 213 simulated animals and treats them as
  exchangeable; tail probabilities near the thresholds are approximate.
- Recovery on synthetic data does not show that any model describes the mice.
- These are development animals. Passing this gate permits a development
  analysis, not confirmation.

## Result — November 24, 2026

**The gate failed, on criterion 6 only.** The run used the frozen plan, 24
datasets of 71 simulated animals plus the resolution check, and was summarised
once. No threshold or setting was changed.

| Criterion | Threshold | Observed | |
|---|---|---|---|
| 1 State, false detection | ≤ 0.05 | 0.000 in `fixed` and `history` | pass |
| 2 State, power | ≥ 0.90 | 1.000 in all four state cells | pass |
| 3 History, false detection | ≤ 0.05 | 0.000 in `fixed`, `drift`, `switching` | pass |
| 4 History, power | ≥ 0.90 | 1.000 in all three history cells | pass |
| 5 Form | right ≥ 0.90, wrong ≤ 0.05 | right 1.000, wrong 0.000 in all four | pass |
| 6 Convergence | ≤ 2% in any cell | 3.9% in `fixed`, 3.4% in `history` | **fail** |
| 7 Numerical resolution | ≤ 2 × 10⁻⁴ | 2.5 × 10⁻⁵ | pass |

All three seeds gave the same decisions in every gated cell.

**Where convergence failed.** Every non-converged fit is the
`switching_history` model, the largest one. It hit the 500-iteration limit for
50 of 213 animals in `fixed` and 43 of 213 in `history`, the two cells that
contain no state, and for 4, 7 and 4 animals in `drift`, `drift_history` and
`drift_history_half`. In the cells that contain switching it always converged.

**Mean held-out gains** (× 10⁻³ nats per trial, 213 animals, 95% interval):

| Cell | `history` − `drift_history` | `history` − `switching_history` | `drift` − `drift_history` | `switching` − `switching_history` | `drift_history` − `switching_history` |
|---|---|---|---|---|---|
| `fixed` | −0.03 [−0.04, −0.02] | −0.70 [−0.78, −0.63] | −0.75 [−0.84, −0.66] | −1.20 [−1.31, −1.09] | −0.67 [−0.75, −0.61] |
| `history` | −0.02 [−0.03, −0.01] | −0.74 [−0.82, −0.67] | 13.20 [11.84, 14.71] | 12.80 [11.42, 14.34] | −0.72 [−0.80, −0.65] |
| `drift` | 12.60 [11.72, 13.47] | 7.25 [6.45, 8.06] | −0.76 [−0.84, −0.67] | −2.16 [−2.37, −1.94] | −5.36 [−5.77, −4.97] |
| `switching` | 10.23 [10.06, 10.40] | 18.65 [18.22, 19.08] | −0.69 [−0.79, −0.60] | −2.47 [−2.68, −2.25] | 8.42 [8.03, 8.80] |
| `drift_history` | 10.44 [9.55, 11.43] | 5.31 [4.57, 6.09] | 12.89 [11.52, 14.27] | 11.29 [9.91, 12.64] | −5.13 [−5.56, −4.70] |
| `switching_history` | 9.82 [9.63, 10.01] | 17.18 [16.71, 17.63] | 11.41 [10.35, 12.53] | 10.66 [9.50, 11.87] | 7.36 [6.97, 7.74] |
| `drift_history_half` | 11.68 [10.73, 12.65] | 6.41 [5.63, 7.25] | 2.96 [2.58, 3.37] | 1.71 [1.29, 2.17] | −5.27 [−5.66, −4.91] |
| `switching_history_half` | 9.94 [9.75, 10.12] | 18.06 [17.61, 18.49] | 2.78 [2.35, 3.24] | 1.41 [0.93, 1.93] | 8.13 [7.73, 8.49] |

**Reported, not gated.**

- *Half-strength history.* The history rule fired in 0.51 of cohorts in
  `drift_history_half` and 0.15 in `switching_history_half`. The rule needs
  both contrasts above δ, and the switching contrast sits near it. History at
  half the development strength would probably be missed.
- *Drift parameters.* Fitted persistence had median 0.995 in the drift cells
  (true 0.995); fitted size had median 0.54–0.57 against a true median of
  0.58–0.63. Where no drift existed, persistence went to its lower bound and
  size towards zero.
- *Best model per animal.* The generating model scored best for 210 of 213
  animals in `drift`, 213 in `switching`, 213 in `drift_history` and 198 in
  `switching_history`. In `fixed` it was best for 130; `drift` was best for 65
  by margins of about 3 × 10⁻⁵, far below δ.
- *Unnecessary models cost little.* A model with a part the data lack scored
  0.02–2.5 × 10⁻³ worse, never better.

**After the result (not planned).** In the two failing cells, the 93
non-converged `switching_history` fits lost 0.63 × 10⁻³ to `history` on
held-out trials and the 333 converged ones lost 0.75 × 10⁻³. This suggests the
optimiser was wandering on a flat surface where the extra states have nothing
to find. It was looked at only after the failure and does not change it.

**What this means.** Under the rule written above, milestone 3 is not frozen
with these settings. The six criteria about telling the alternatives apart were
met with wide margins, so the comparison itself looks sound at this depth and
at full development strength. What is not shown is that the largest model can
be fitted reliably when no state is present, which is one of the situations the
real data may be in. A new dated protocol must change the fitting of
`switching_history`, state the change before running, and repeat the check.

The run was paused twice by the computer sleeping. Each dataset is written only
when complete and no result depends on timing.

Portable result: [`results/within_animal_recovery_v1.json`](results/within_animal_recovery_v1.json).

**Follow-up.** [Revision 1](WITHIN_ANIMAL_RECOVERY_REVISION_PROTOCOL.md) raised
the iteration limit to 5,000 under a new protocol and passed all seven
criteria. The result above is unchanged and stands as recorded.

## Reproduction

```bash
python -m scripts.recover_within_animal --stage plan
python -m scripts.recover_within_animal --stage run
python -m scripts.recover_within_animal --stage summarize
```
