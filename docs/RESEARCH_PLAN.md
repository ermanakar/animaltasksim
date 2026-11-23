# Research plan — November 2026: slow state before trial history

This plan replaces the [September 2026 plan](archive/RESEARCH_PLAN_2026-09.md),
which is archived unchanged. Completed results keep their own dated reports.

## Why the direction changes

The September plan asked whether previous stimulus strength modulates the
effect of the previous outcome on the next choice. That question has been taken
as far as the current data allow:

- The effect replicated on a reserved cohort, and it is small
  ([result](REPLICATION_RESULT.md)).
- It shrinks under longer-history and side-bias controls
  ([competing history](COMPETING_HISTORY.md), [bias controls](BIAS_HISTORY.md)).
- Split by outcome, most of its reward half disappears under those controls
  ([outcome split](OUTCOME_SPLIT.md)).
- Generative models could not separate trial history from slow drift or state
  switching, and the recovery study failed its gate. The report named more
  within-animal data as the remedy ([generative comparison](GENERATIVE_COMPARISON.md)).

Two things are new. First, the [deep development reference](DEEP_REFERENCE_PROTOCOL.md)
now holds 21 to 103 sessions per mouse for 72 mice, against about 1.4 before.
Second, the [first look](DEEP_EXPLORATION.md) at it suggests that the largest
structure in the behaviour is slow: side preference lasting many trials, stable
differences between animals, and within-session changes in speed. Trial-to-trial
outcome effects are small beside it.

Continuing to refine a 10⁻³ nats/trial effect while the dominant structure is
unmodelled is the wrong order of work.

## Question

**How much of a mouse's choice behaviour in this task is explained by slowly
changing latent state, and what remains for trial history once that state is
modelled within each animal?**

"State" is a hypothesis about structure in the data, not an assumed fact.
Mice alternating between engaged and biased states in this task has been
reported before ([Ashwood et al., 2022](https://www.nature.com/articles/s41593-021-01007-z));
the project makes no novelty claim for it.

## Explicit alternatives

Each must be fitted with the same information access and scored on the same
held-out sessions of the same animals.

1. **Fixed animal.** A per-animal psychometric curve and constant side bias.
2. **Fixed animal plus trial history.** Adds outcome-dependent choice history.
3. **Continuous drift.** Side bias and sensitivity change smoothly in time.
4. **Discrete states.** The animal switches between a few response modes.
5. **Drift or states, plus trial history.**

The exploration favours 3–5 over 1–2, but it fitted none of them. A result in
which model 2 predicts held-out sessions as well as 3–5 would refute the new
direction, and would be reported as such.

## Milestones and gates

Each milestone has a written protocol before its analysis runs. A milestone
that fails its gate stops the sequence; the failure is the result.

1. **Convert the deep reference.** Turn raw arrays into schema-valid trials per
   session family, reusing the verified wheel-choice coding and omission
   handling. Eligibility and quality rules are written before any outcome is
   examined and never depend on history or state measures.
   *Gate:* converted biased sessions reproduce the source-reconciled reference
   trial for trial on the sessions they share.

2. **Recovery at the new data depth.** Repeat the generative recovery study with
   the real per-animal session counts and task schedules, synthetic responses,
   and the effect sizes observed in development. Include a fitted continuous
   drift model, which the earlier study lacked.
   *Gate:* predeclared false-detection and power thresholds are met for telling
   alternatives 2, 3, 4 and 5 apart. If they are not, no claim about which
   model is right can be made from these data.

3. **Within-animal comparison on development mice.** Fit each alternative per
   animal on a fixed subset of its sessions and score the remaining sessions.
   The endpoint, the session split, the number of states and all priors are
   fixed in advance.
   *Gate:* the endpoint and decision rule are frozen and hashed before fitting.

4. **Predictions that choice models did not use.** If state models win, test
   what they predict about measures they were not fitted to: response time and
   omissions by inferred state, clustering of easy-trial errors, and state
   occupancy across a session and across training.
   *Gate:* each prediction is written with its direction before it is checked.

5. **Confirmation on untouched animals.** Animals in neither the development
   set nor the completed replication cohort remain unlisted and undownloaded.
   A confirmation protocol with sample size, eligibility and stopping rules is
   frozen before any of them is accessed. Development animals cannot confirm
   anything.

6. **Consequences for agents.** Only after milestone 3: define state-level
   fingerprints and ask whether any agent reproduces them. Until then no
   architecture is changed to target a state-level result.

## Decision rules

- Advance a richer model only if it improves the frozen held-out measure, the
  animal-level uncertainty supports the improvement, and recovery shows the
  comparison can distinguish it from simpler alternatives.
- State the practically meaningful difference before the comparison, on
  development data only.
- If the simpler model predicts as well, report that and keep the simpler model.
- A failed prediction is recorded with the same prominence as a successful one.
  The [outcome split](OUTCOME_SPLIT.md) is the current example.

## Standing rules

- Every analysis is labelled exploratory or confirmatory before it runs.
- Existing run directories, the adopted reference, the reconciled candidate and
  the frozen replication are never overwritten. New work goes to new paths.
- Inputs, code and plans are hashed; portable snapshots of every reported
  number go under `docs/results/`.
- Session families are not pooled without a stated reason.
- Software lesions and model fits do not establish mechanisms in animals.

## Open items carried forward

- The generative refit without optogenetic and RPE sessions had not finished on
  November 23 ([sensitivity check](VARIANT_SENSITIVITY.md)).
- The adopted reference still contains those 16 variant sessions. Rebuilding it
  is a dated provenance decision, to be taken with milestone 1.
- 21 of the 45 replication animals come from one lab; cross-lab generalization
  of the replicated effect remains untested.

## Deferred

The evidence-dependent history effect is not pursued further on its own; it
returns as a term inside alternative 5. Scalar sweeps, new control circuits,
the adaptive controller, PRL transfer and DMS expansion stay paused. Existing
CLI arguments, schema keys, paths and defaults remain stable, and historical
scripts stay in place.

## Scientific basis

[The IBL task paper](https://elifesciences.org/articles/63711) separates
sensory, choice-history and block contributions.
[Ashwood et al.](https://www.nature.com/articles/s41593-021-01007-z) describe
discrete engaged and biased states in the same task.
[Roy et al.](https://pubmed.ncbi.nlm.nih.gov/33412101/) describe continuous
drift of psychophysical weights during learning.
[Lak et al.](https://elifesciences.org/articles/49834) motivate
difficulty-dependent outcome effects and the following-trial control.
[Wilson and Collins](https://elifesciences.org/articles/49547) motivate
recovery checks before interpretation.
