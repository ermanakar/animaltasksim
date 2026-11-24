# Within-animal recovery, revision 1 — November 24, 2026

The [recovery study](WITHIN_ANIMAL_RECOVERY_PROTOCOL.md) failed its gate on
convergence: the `switching_history` model reported non-convergence in 3.9% and
3.4% of fits in the two cells without a state, against a limit of 2%. Its rule
was that a new dated protocol must change the fit, state the change before
running, and repeat the check. This is that protocol.

It was written after the pilot described below and before any fit of the
revised study was run.

## What was already seen

- The full result of the first study, including every gain and probability.
- **A pilot on seed 3**, which is not one of the study seeds (0, 1, 2). The
  first study stored whether a fit converged but not why it stopped, and its
  report said the failing fits had reached the iteration limit without having
  checked. The pilot simulated every third animal (24) in the `fixed` and
  `history` cells with seed 3 and fitted `switching_history` on the
  even-numbered sessions from both starts, at limits of 500 and 5,000:
  - at 500, 9 of 48 fits in `fixed` and 12 of 48 in `history` stopped with
    "total number of iterations reached limit"; every other fit stopped on the
    relative-reduction tolerance. No other stop reason occurred;
  - at 5,000, all 96 fits converged, the longest in 1,069 iterations.

  The statement in the first report is therefore supported, on a different
  seed. The pilot informed one choice only: the new limit.

## The change

The iteration limit of the optimiser is raised from 500 to **5,000** for every
model. Nothing else changes: models, penalties, starts, tolerances, state grid,
scoring, generators, seeds, endpoints, δ, cohort resampling, the seven criteria
and their thresholds are those of the first protocol.

## What is refitted and what is carried over

The optimiser is deterministic. A fit that stopped on tolerance below 500
iterations is identical under a higher limit.

- **Carried over** from the first study: `fixed`, `history`, `drift` and
  `drift_history`, in which every fit reported convergence, and the 81-state
  resolution check, which involves only the drift models. This is checked, not
  assumed: the run stops if any carried fit in the first study was not
  converged, and all six models are refitted under the new limit on the seed-0
  `fixed` dataset, where the four carried models must reproduce their earlier
  scores to within 10⁻¹² nats per trial.
- **Refitted** on all 24 datasets: `switching` and `switching_history`. The
  switching models keep the better of two starts, so a start that had not
  finished could change the selected fit even where the selected fit had
  converged.

The synthetic choices are regenerated from the same seeds and must match the
first study animal for animal (generating parameters and committed-trial
counts) or the run stops.

## Gate

The seven criteria of the first protocol, unchanged, applied once to the
combined scores. All six detection and discrimination criteria are recomputed,
because the scores of both switching models may change. The earlier passes are
not carried over as passes.

- **Pass:** milestone 3 may be frozen with these six models, these endpoints
  and the 5,000-iteration limit.
- **Fail:** the failing criterion is the finding, as before. A further revision
  would need a new dated protocol; this one is not amended.

Also reported: how many selected fits changed, the largest change in any
animal's held-out score, and the same descriptive tables as the first study.

## Limits

- The new limit was chosen after seeing the first result and a pilot. The pilot
  used a seed outside the study, but the animals' schedules are the same.
  Convergence is a numerical property, not an effect estimate, so this is a
  smaller concern than tuning on an outcome would be; it is still disclosed.
- Reaching tolerance shows the optimiser stopped, not that it found the best
  solution. Two starts is a small number for a model with three states.
- Every limit of the first protocol still applies. Recovery on synthetic data
  does not show that any model describes the mice.

## Result — November 24, 2026

**The gate passed, on all seven criteria.** The run used the frozen plan and
was summarised once.

| Criterion | Threshold | Observed | |
|---|---|---|---|
| 1 State, false detection | ≤ 0.05 | 0.000 | pass |
| 2 State, power | ≥ 0.90 | 1.000 | pass |
| 3 History, false detection | ≤ 0.05 | 0.000 | pass |
| 4 History, power | ≥ 0.90 | 1.000 | pass |
| 5 Form | right ≥ 0.90, wrong ≤ 0.05 | right 1.000, wrong 0.000 | pass |
| 6 Convergence | ≤ 2% in any cell | 0.0% in every cell | pass |
| 7 Numerical resolution | ≤ 2 × 10⁻⁴ | 2.5 × 10⁻⁵ (carried over) | pass |

**Checks on the design.** The four carried-over models reproduced their
earlier scores exactly on the check dataset (largest difference 0). The
regenerated synthetic data matched the first study for all 1,704 simulated
animals.

**What changed.** All 108 previously non-converged `switching_history` fits
converged. Its held-out score changed for 59 animals in `fixed`, 52 in
`history`, 5 in `drift`, 7 in `drift_history` and 4 in `drift_history_half`,
by at most 6.3 × 10⁻⁴ nats per trial for any animal. That is more animals than
had failed to converge, because an unfinished second start sometimes overtook
the selected one, as anticipated. `switching` changed for one animal, by
2 × 10⁻⁸. Detection probabilities are the same as in the first study to the
reported precision, including the two ungated half-strength cells (history
detected in 0.51 and 0.15 of cohorts).

Mean gains involving `switching_history` in the two cells that had failed
(× 10⁻³ nats per trial):

| Cell | `history` − `switching_history` | `switching` − `switching_history` | `drift_history` − `switching_history` |
|---|---|---|---|
| `fixed` | -0.70 [-0.78, -0.63] | -1.20 [-1.31, -1.08] | -0.67 [-0.75, -0.61] |
| `history` | -0.74 [-0.82, -0.67] | 12.80 [11.42, 14.34] | -0.72 [-0.80, -0.66] |

**What this means.** Milestone 3 may be frozen with these six models, these
endpoints and the 5,000-iteration limit. The limits above stand: history at
half the development strength would probably be missed, the number of starts
is small, and nothing here is about real mice.

Portable result: [`results/within_animal_recovery_v2.json`](results/within_animal_recovery_v2.json).

## Reproduction

```bash
python -m scripts.recheck_within_animal_recovery --stage plan
python -m scripts.recheck_within_animal_recovery --stage run
python -m scripts.recheck_within_animal_recovery --stage summarize
```
