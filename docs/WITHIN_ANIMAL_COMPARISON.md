# Within-animal comparison on development mice — November 24, 2026

Milestone 3 of the [research plan](RESEARCH_PLAN.md), run under the
[protocol](WITHIN_ANIMAL_COMPARISON_PROTOCOL.md) frozen beforehand. The
protocol file is unchanged since the plan was frozen (its hash is recorded in
the plan), so the result is reported here.

**Status: exploratory.** Development animals; no claim is confirmed.

## Short version

On held-out sessions of the same 71 mice:

- A slowly changing latent state improves prediction **beyond** trial history.
- Trial history improves prediction **beyond** the state.
- Whether the state is better described as drifting or switching is
  **unresolved**; drift scores slightly better on average.
- State and history mostly explain the **same** variation. Either one alone
  recovers about four fifths of what both recover together.

All three decisions were the same in both planned sensitivity analyses.

## Run conditions

| Condition | Limit | Observed | |
|---|---|---|---|
| Selected fits not converged | ≤ 2% | 0 of 426 | met |
| Change in drift scores with 81 states | ≤ 2 × 10⁻⁴ | 7.2 × 10⁻⁵ | met |

71 animals, 2,089 sessions, 1,518,119 committed trials.

## Endpoints

Mean gain in held-out score, × 10⁻³ nats per trial, with 95% bootstrap interval
over animals. δ = 1 in these units.

| Endpoint | Contrast | Gain | Animals with a gain | Decision |
|---|---|---|---|---|
| State beyond history | `history` − `drift_history` | 14.16 [12.33, 16.11] | 71 of 71 | **detected** |
| | `history` − `switching_history` | 12.33 [10.65, 14.08] | 69 of 71 | |
| History beyond state | `drift` − `drift_history` | 10.92 [9.67, 12.19] | 71 of 71 | **detected** |
| | `switching` − `switching_history` | 12.48 [10.77, 14.20] | 71 of 71 | |
| Form of the state | `drift_history` − `switching_history` | −1.83 [−3.19, −0.48] | 29 of 71 | **unresolved** |

The form interval lies below zero, favouring drift, but its upper bound is not
below −δ, so the rule does not name a form.

## How much each part explains

Mean held-out loss per trial, and the same number as the typical probability a
model gave to the choice the mouse actually made:

| Model | Loss (nats per trial) | Typical probability of the actual choice |
|---|---|---|
| `fixed` | 0.4077 | 66.5% |
| `history` | 0.3558 | 70.1% |
| `switching` | 0.3560 | 70.0% |
| `drift` | 0.3526 | 70.3% |
| `switching_history` | 0.3435 | 70.9% |
| `drift_history` | 0.3417 | 71.1% |

| Quantity (× 10⁻³) | Gain |
|---|---|
| History alone over `fixed` | 51.87 [44.94, 59.85] |
| Drift alone over `fixed` | 55.11 [47.48, 64.17] |
| Switching alone over `fixed` | 51.72 [44.33, 60.41] |
| Share of the history gain remaining after drift | 0.21 [0.18, 0.24] |
| Share of the history gain remaining after switching | 0.24 [0.20, 0.28] |

Arithmetic on the means above (not a separate planned estimate): drift and
history together gain 66.0 over `fixed`. Of that, 14.2 is unique to the state,
10.9 is unique to history, and about 41 can be obtained from either. The data
do not say which of the two the shared part belongs to.

`drift_history` was the best-scoring model for 42 animals and
`switching_history` for 29. No simpler model was best for any animal.

## Sensitivity analyses

| | State (drift) | State (switching) | History (drift) | History (switching) | Form | Decisions |
|---|---|---|---|---|---|---|
| Primary, 71 animals | 14.16 [12.33, 16.11] | 12.33 [10.65, 14.08] | 10.92 [9.67, 12.19] | 12.48 [10.77, 14.20] | −1.83 [−3.19, −0.48] | state, history, unresolved |
| Chronological split | 15.58 [13.13, 18.28] | 12.84 [10.68, 15.19] | 9.90 [8.70, 11.11] | 12.24 [10.22, 14.36] | −2.74 [−4.84, −0.60] | same |
| Without the largest lab, 38 animals | 12.48 [10.15, 15.10] | 11.52 [9.46, 13.72] | 13.09 [11.36, 14.78] | 15.07 [12.69, 17.51] | −0.97 [−2.86, 0.88] | same |

Every lab's mean gain is positive for both state contrasts and both history
contrasts (nine labs, one to 33 animals each; values in the result file). The
form contrast changes sign between labs.

## Fitted drift

Median [quartiles] across animals:

| Model | Persistence per trial | Stationary size (log-odds) |
|---|---|---|
| `drift` | 0.964 [0.931, 0.980] | 1.55 [1.33, 1.79] |
| `drift_history` | 0.9966 [0.9918, 0.9979] | 0.82 [0.71, 1.04] |

Without history terms the drift is fast, with a time constant near 30 trials,
and large: it appears to stand in for history. With history terms it is slow,
with a time constant near 300 trials. This is a description of fitted
parameters, not a tested claim.

## What this does and does not show

It supports the direction adopted on November 23: a slow state adds predictive
value within animals that trial history does not supply, in every animal for
the drift contrast. It does **not** support the stronger reading that trial
history is merely a shadow of slow state. History keeps a gain of about the
same size once state is modelled, also in every animal.

Read against the predeclared outcomes, this is "state detected, history
detected": both contribute, and the size of each is reported.

Limits, in addition to those in the protocol:

- **The shared part is ambiguous.** A fast drift and a history rule can
  generate nearly the same choices. Held-out choice likelihood cannot separate
  them; milestone 4 exists for that.
- **"State" is not yet "mood".** The latent variable is a side bias that
  wanders. It could reflect engagement, a motor or postural preference, or the
  animal's own lagging estimate of the block, which the models represent only
  by one fixed block term. This limit was recognised after seeing the result.
- **Effect sizes are near the strengths the recovery study validated**, which
  is where its rules are known to work. That study does not show the rules
  behave well for behaviour unlike its generators.
- Development animals, exploratory status. Nothing here is confirmed.

## Files

- Result: [`results/within_animal_comparison_v1.json`](results/within_animal_comparison_v1.json)
- Per-animal scores: [`results/within_animal_comparison_v1_subject_scores.csv`](results/within_animal_comparison_v1_subject_scores.csv)
- Protocol: [WITHIN_ANIMAL_COMPARISON_PROTOCOL.md](WITHIN_ANIMAL_COMPARISON_PROTOCOL.md)
