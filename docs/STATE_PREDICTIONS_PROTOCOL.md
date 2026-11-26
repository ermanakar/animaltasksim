# Predictions from the latent state — protocol, November 25, 2026

Milestone 4 of the [research plan](RESEARCH_PLAN.md). The
[within-animal comparison](WITHIN_ANIMAL_COMPARISON.md) found that a slowly
wandering side bias improves the prediction of held-out choices. Choices alone
cannot say what that variable is. This study asks what it predicts about two
measures the models were never fitted to, response time and omitted trials,
and whether part of it is the animal's own lagging estimate of the block.

**Status: exploratory.** Development animals. Nothing here can confirm a claim.

This document was written before any response time or omission was related to
a fitted state. The analysis code was run only on synthetic data.

The code was developed on invented schedules with synthetic choices. One
measure changed during that development: the first version of P3 compared each
omission with committed trials in the same session and 100-trial bin, and it
removed a true effect in synthetic data, because the state changes little
within 100 trials. The version below replaced it before any real data were
read.

## What was already seen

- Everything in the comparison report, including the fitted drift parameters.
- The [first look](DEEP_EXPLORATION.md): response time rises by about 40% over
  a session, and choices follow a block change within about five trials.
- Overall counts from the conversion manifest: 0.37% of trials in these
  sessions are omissions, and 136 of the 2,089 sessions have a response time
  for fewer than half of their choices.
- No relation between any fitted state and response time, omissions or block
  changes has been computed.

## State estimate

One model is used: `drift_history`, the best-scoring model in the comparison,
with the settings frozen there. For each animal it is fitted on even-numbered
sessions and applied to odd-numbered ones, then the reverse.

The estimate for trial *t*, written b̂ₜ, is the model's expected side bias
before the choice on trial *t*: it uses the fitted parameters from the other
sessions and only the choices on **earlier** trials of the same session.
Positive means towards the right. It is in log-odds. The switching model is not
analysed here; its form was unresolved and adding it would double the number of
tests.

## Predictions

Written with their direction before any is checked.

| | Prediction | Measure | Direction |
|---|---|---|---|
| **P1** | A choice towards the side the state favours is faster than a choice against it | Coefficient of choice × b̂ on log response time | negative |
| **P2** | The more strongly the state leans to either side, the slower the response | Coefficient of \|b̂\| on log response time | positive |
| **P3** | Omitted trials occur when the state leans more strongly | Adjusted \|b̂\| on omitted trials minus committed trials | positive |
| **D1** | Just after the block changes side, the state still leans towards the old side | b̂ signed towards the new side, trials 1–10 after a change | negative |

P1 is what a bias inside the decision process predicts. P2 and P3 are what
disengagement predicts. D1 is a diagnostic of the alternative noted after the
comparison: the models give the block one fixed term that changes instantly,
so an animal that updates gradually would leave its lag in the latent bias.

The project has no strong prior for P2 or P3. They are stated in the direction
that the engagement reading requires, so that a null or opposite result counts
against that reading.

## Measures

**Response time (P1, P2).** Committed trials with a recorded response time, in
sessions where at least half of the committed trials have one. Response time is
clipped to 50 ms – 60 s and logged. Per animal, an ordinary least-squares
regression with every variable centred within session, on: choice × b̂, |b̂|,
and these controls: choice side, correct or not, absolute contrast (four
indicator terms), position in the session in 100-trial bins (capped at 900+),
choice × the model's history log-odds, and choice × the model's block
log-odds. Animals with fewer than 500 usable trials are left out.

**Omissions (P3).** A trial is usable if the five trials before it were all
committed, so that the estimate rests on comparable information. Within each
animal, |b̂| on usable trials is centred within session and adjusted for
position in the session (the same 100-trial bins) by least squares. The
animal's value is the mean adjusted |b̂| on omitted trials minus the mean on
committed trials. Animals with fewer than ten usable omissions are left out.

**Block lag (D1).** Changes between a 0.2 and a 0.8 block. b̂ multiplied by the
sign of the new side, averaged over trials 1–10 after each change, per animal.
The same quantity for trials 31–50 is reported for comparison.

## Decision rule

Each quantity is an equal-animal mean with a 95% bootstrap interval over
animals (10,000 resamples). The smallest sizes of interest are fixed here:

- P1, P2: 0.02 in log response time per unit of log-odds (about 2%);
- P3: 0.02 log-odds;
- D1: 0.05 log-odds.

These were set on synthetic data only. During development, a true omission
effect of 1.0 log-odds per unit of |b| appeared as a difference of about 0.05
in adjusted |b̂|, because b̂ is a shrunken estimate; 0.02 therefore corresponds
to an effect a little under half that size. A true response-time coefficient
of 0.06 on |b| was recovered as about 0.03.

A prediction is **supported** if the whole interval lies beyond the size of
interest in the predicted direction, **opposite** if it lies beyond it in the
other direction, and otherwise **not supported at the size of interest**, with
the estimate reported. Four tests, no multiplicity correction.

## Method check on synthetic animals

This analysis is new, and b̂ is built from choices, so an artefact is possible.
Before any real response time is read, the full analysis runs on two sets of 71
synthetic animals on the real schedules:

- **Null.** Drift-plus-history choices; response time depends on contrast,
  position in the session, choice side and correctness, but not on the state;
  omissions occur at random at a rate of 0.4%; the block acts instantly.
- **Effect.** The same, with log response time changed by −0.06 × choice × b
  and +0.06 × |b|, and the log-odds of an omission raised by 1.0 × |b|, where
  b is the true latent bias.

**Gate:** in the null set, P1, P2, P3 and D1 are all "not supported at the size
of interest"; in the effect set, P1, P2 and P3 are all "supported". If the gate
fails, the real analysis does not run under this protocol. The estimated
coefficients in the effect set are reported, since b̂ is a noisy estimate of b
and recovered sizes may be smaller than the true ones.

## What each outcome will be taken to mean

- **P2 or P3 supported:** the latent bias carries information about response
  vigour or task engagement beyond choices. Not a demonstration of a named
  internal state.
- **Only P1 supported:** the variable behaves like a bias in the decision
  process, with no sign that it reflects engagement.
- **None supported:** the variable is, so far, a description of choices only.
- **D1 supported:** part of the latent bias is the animal's lagging block
  estimate. The comparison's "state beyond history" gain is then partly a gain
  from modelling block learning, and the next model must represent that
  explicitly before "state" is interpreted further.
- **Any "opposite":** reported with the same prominence.

## Limits

- Development animals and exploratory status, as before.
- A linear regression with a chosen set of controls. Other controls could
  change the estimates; none will be tried under this protocol.
- b̂ is a noisy, one-sided estimate, weakest early in a session and after
  omissions.
- Omissions are rare (about 0.4% of trials), so P3 rests on few events per
  animal and on the animals that have at least ten.
- Response time here is the time of the recorded response, which mixes
  decision and movement.
- Associations do not show that the state causes slower responses or
  omissions; a third factor such as satiety could drive both.

## Reproduction

```bash
python -m scripts.predict_from_state --stage plan
python -m scripts.predict_from_state --stage check
python -m scripts.predict_from_state --stage run
python -m scripts.predict_from_state --stage summarize
```
