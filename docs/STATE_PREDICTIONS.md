# Predictions from the latent state — result, November 26, 2026

Milestone 4 of the [research plan](RESEARCH_PLAN.md), run under the
[protocol](STATE_PREDICTIONS_PROTOCOL.md) frozen beforehand. The protocol file
is unchanged since the plan was frozen (its hash is recorded in the plan), so
the result is reported here.

**Status: exploratory.** Development animals; no claim is confirmed. These are
associations, not causes.

## Short version

The slowly wandering side bias found in the
[within-animal comparison](WITHIN_ANIMAL_COMPARISON.md) was estimated from
choices only. It also predicts two things it was never fitted to:

- **P1 supported.** Choices towards the side the state favours are faster.
- **P2 supported.** The more strongly the state leans to either side, the
  slower the response.
- **P3 supported.** The state leans more strongly just before omitted trials.
- **D1 not supported at the size of interest.** Just after a block change the
  state does lean slightly towards the old side, but the lag is about half the
  size fixed in advance as worth interpreting.

All four readings were the same without the largest lab.

## Method check on synthetic animals

Run before any real response time was read. 71 synthetic animals per set on
the real schedules. **Gate passed.**

| | Null set (nothing planted) | Effect set (planted) |
|---|---|---|
| P1 | −0.0018 [−0.0041, 0.0006], not supported | −0.0594 [−0.0621, −0.0567], supported (true −0.06) |
| P2 | 0.0000 [−0.0055, 0.0062], not supported | 0.0318 [0.0244, 0.0397], supported (true 0.06) |
| P3 | −0.0001 [−0.0066, 0.0066], not supported | 0.0416 [0.0307, 0.0540], supported |
| D1 | −0.0014 [−0.0035, 0.0007], not supported | −0.0007 [−0.0027, 0.0012], not supported (none planted) |

Building the estimate from choices did not by itself create any of the four
associations. P2 was recovered at about half its true size.

## Result on development mice

71 animals, no fit failed to converge. 1,412,034 trials with a usable response
time, 2,733 usable omissions, 24,969 block changes. Equal-animal means with
95% bootstrap intervals over animals.

| | Measure | Predicted | Size of interest | Estimate | Animals in predicted direction | Reading |
|---|---|---|---|---|---|---|
| **P1** | choice × b̂ on log response time | negative | 0.02 | −0.088 [−0.107, −0.070] | 61 of 71 | **supported** |
| **P2** | \|b̂\| on log response time | positive | 0.02 | 0.133 [0.104, 0.161] | 62 of 71 | **supported** |
| **P3** | adjusted \|b̂\|, omitted − committed | positive | 0.02 | 0.108 [0.070, 0.146] | 50 of 57 | **supported** |
| **D1** | b̂ towards the new side, trials 1–10 | negative | 0.05 | −0.024 [−0.032, −0.018] | 59 of 71 | **not supported at the size of interest** |

P3 uses the 57 animals with at least ten usable omissions. For D1, the same
quantity on trials 31–50 after a change is +0.023.

In plain terms: per unit of log-odds of estimated bias, a choice with the bias
is about 8% faster than average and one against it about 8% slower; and a
response is about 14% slower per unit of bias strength in either direction.

### Without the largest lab (38 animals)

| | Estimate | Reading |
|---|---|---|
| P1 | −0.078 [−0.099, −0.056] | supported |
| P2 | 0.144 [0.110, 0.178] | supported |
| P3 (29 animals) | 0.113 [0.056, 0.164] | supported |
| D1 | −0.037 [−0.047, −0.027] | not supported at the size of interest |

## What this does and does not show

Read against the outcomes written beforehand:

- **P2 and P3 supported:** the latent bias carries information about response
  vigour and task engagement beyond choices. This is not a demonstration of a
  named internal state.
- **P1 supported as well:** the variable also behaves like a bias inside the
  decision process. The two readings are not exclusive, and this study does
  not separate them.
- **D1:** a lag exists (the interval excludes zero and 59 of 71 animals show
  it), but it is small: about 0.02 log-odds against a typical bias size of
  about 0.8. Under the frozen rule it is not large enough to say that the
  "state beyond history" gain is partly block learning. The concern is reduced,
  not removed.

Limits, in addition to those in the protocol:

- **The estimate is built from recent choices.** A run of same-side choices
  raises |b̂|. P2 and P3 may therefore mean "the animal is slower and more
  likely to miss a trial during one-sided runs". That is still a finding about
  engagement, but it does not require a hidden variable separate from the
  choice pattern itself. The synthetic null shows this is not an artefact of
  the method; it does not show which description is right.
- **Sizes are imprecise.** The synthetic check recovered a planted P2 effect
  at half size; real sizes may differ from the estimates in either direction
  because the regression controls are a fixed, chosen set.
- **Omissions are rare.** P3 rests on 2,733 events and excludes 14 animals.
- **D1 only measures a lag in the latent bias.** A slow block update could
  also sit in the history terms, which this study did not examine.
- Development animals, exploratory status. Nothing here is confirmed.

## Files

- Result: [`results/state_predictions_v1.json`](results/state_predictions_v1.json)
- Method check: [`results/state_predictions_v1_check.json`](results/state_predictions_v1_check.json)
- Per-animal values: [`results/state_predictions_v1_subject_values.csv`](results/state_predictions_v1_subject_values.csv)
- Protocol: [STATE_PREDICTIONS_PROTOCOL.md](STATE_PREDICTIONS_PROTOCOL.md)
