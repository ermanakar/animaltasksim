# Within-animal comparison protocol — November 24, 2026

Milestone 3 of the [research plan](RESEARCH_PLAN.md): the first fit of the
within-animal models to real choices. It asks how much of each development
mouse's choice behaviour a slowly changing latent state explains on held-out
sessions, and what remains for trial history once that state is modelled.

**Status: exploratory.** These are development animals that the project has
already examined. Nothing here can confirm a claim; confirmation is milestone 5,
on animals not yet accessed.

This document was written before any within-animal model was fitted to a real
choice. The code was exercised only on synthetic choices.

## What was already seen

- The [first look](DEEP_EXPLORATION.md) at these animals: side preference
  lasting many trials, stable differences between animals, slow changes within
  sessions. It fitted no state model.
- The [outcome split](OUTCOME_SPLIT.md) and
  [variant sensitivity](VARIANT_SENSITIVITY.md) on 83 development animals,
  where a slowly updated bias term removed most of a history effect, and
  cross-animal models with fixed drift or switching scored better than history
  alone.
- The [recovery study](WITHIN_ANIMAL_RECOVERY_PROTOCOL.md) and its
  [revision](WITHIN_ANIMAL_RECOVERY_REVISION_PROTOCOL.md), on synthetic choices.

The expectation from these is that state models will score better than history
alone. That expectation is the reason the decision rules below were fixed by
the recovery study and are not chosen now.

## Data

Every converted biased-family session with at least 150 trials, for animals
with at least six such sessions: 71 animals and 2,089 sessions, the set whose
schedules the recovery study used. All are standard `biasedChoiceWorld`
sessions; none is an optogenetic or reward-variant session. No
performance-based filter is applied. Training and ephys sessions are not used.

## Models, fitting and scoring

Exactly as validated in revision 1 of the recovery study: the six models
`fixed`, `history`, `drift`, `drift_history`, `switching` and
`switching_history`; ridge penalty 1; 41 drift states; three switching states
with two deterministic starts; latent states restart at each session; an
iteration limit of 5,000. Models are fitted per animal on even-numbered
sessions and scored on odd-numbered ones, then the reverse. The score is mean
negative log likelihood per committed trial, per animal. No setting is tuned on
these data.

## Endpoints and decision rules

Unchanged from the recovery protocol. Each endpoint is an equal-animal mean of
a per-animal difference in held-out score with a 95% bootstrap interval over
animals (10,000 resamples). δ = 0.001 nats per trial.

| Endpoint | Contrasts | Rule |
|---|---|---|
| **State** beyond trial history | `history` − `drift_history`; `history` − `switching_history` | detected if the lower bound exceeds δ for at least one |
| **History** beyond state | `drift` − `drift_history`; `switching` − `switching_history` | detected if the lower bound exceeds δ for both |
| **Form** of the state | `drift_history` − `switching_history` | switching if the lower bound exceeds δ; drift if the upper bound is below −δ; otherwise unresolved |

Three endpoints are tested with fixed rules and no multiplicity correction.

## What each outcome will be taken to mean

- **State not detected.** History alone predicts held-out sessions as well as
  the state models. The direction adopted on November 23 is refuted on these
  animals and will be reported as refuted.
- **State detected, history detected.** Both contribute within animals. The
  size of each is reported; neither is called a mechanism.
- **State detected, history not detected.** Trial history adds less than δ once
  state is modelled, *or* its remaining effect is too small for this design:
  the recovery study detected history at half the development strength in only
  0.51 and 0.15 of cohorts. The estimate and interval are reported and absence
  is not claimed.
- **Form.** Read only if state is detected. "Unresolved" is a result.

Also reported, without a decision rule: `fixed` − `history`,
`fixed` − `drift` and `fixed` − `switching`; the share of the history gain that
remains after state, (`drift` − `drift_history`) / (`fixed` − `history`) and
the same for switching, with bootstrap intervals; the number of animals with a
positive gain for every contrast; the best-scoring model per animal; fitted
drift persistence and size; and mean gains by lab.

## Conditions on the run

Stated now, checked at the summary:

1. At most 2% of selected fits report non-convergence.
2. Refitting both drift models with 81 states changes held-out scores by at
   most 2 × 10⁻⁴ nats per trial on average across animals.

If either fails, the endpoints are reported as numbers and **no decision is
drawn** from them.

## Planned sensitivity analyses

Their purpose is to show whether the primary reading depends on a choice made
here. They do not replace it.

- **Chronological split.** Fit on the earlier half of each animal's sessions
  and score the later half, then the reverse. Behaviour may change across
  weeks; alternating sessions hides that, this split exposes it.
- **Without the largest lab.** 33 of the 71 animals come from one lab. The
  endpoints are recomputed on the other 38, with no refitting.

If a sensitivity analysis gives a different decision on an endpoint, that
endpoint is reported as not robust.

## Outputs

`plan` freezes the inputs, code and this document by hash. `run` writes scores
and prints nothing about them. `summarize` computes everything above once and
refuses to overwrite its result. Portable copies go under `docs/results/`.

## Limits

- Development animals, exploratory status, and an expectation formed from the
  same animals.
- Held-out sessions come from the same animals as the fitted ones. The result
  describes prediction within these mice, not other mice.
- Latent states restart each session, so structure lasting across sessions is
  invisible to every model here.
- The models are a small family. Lapses, several slow processes at once, a
  different number of states, and changes in sensitivity under drift are not
  represented. A better score for one model does not show that the mouse has
  that state.
- The recovery study validated these rules for generators of development
  strength on these schedules. Smaller effects, and real behaviour unlike any
  generator, are outside what it showed.
- 46% of the animals are from one lab.

## Reproduction

```bash
python -m scripts.compare_within_animal --stage plan
python -m scripts.compare_within_animal --stage run
python -m scripts.compare_within_animal --stage summarize
```
