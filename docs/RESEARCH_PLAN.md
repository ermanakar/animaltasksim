# Research plan — September 2026 reset

## Question and alternatives

Does previous stimulus strength modulate the influence of a rewarded or
unrewarded choice on the next decision, beyond present stimulus and block
context? Contrast is a proxy for available evidence, not subjective uncertainty.

The current reference gives a negative raw weak-minus-strong failure retry gap.
The earlier controller was designed to produce a positive gap. Neither changing
that target after seeing results nor adding circuits establishes an explanation.

Compare explicit alternatives:

1. Current stimulus and block context account for the apparent signature.
2. Outcome-dependent choice history adds predictive information.
3. Prior evidence further modulates that history, with the direction estimated
   from data rather than hard-coded.
4. A simple learning or inference model explains the same effect as well as the
   adaptive controller, making its extra machinery unnecessary for this task.

## Completed reset

- Corrected psychometric/history omission handling; regression tests cover it.
- Corrected `max_sessions` to count actual sessions before chunking. Historical
  runs are not regenerated; identical old commands can now train on more data
  and take longer. Re-running is a new experiment, not byte-for-byte reproduction.
- Added a fixed exploratory session-held-out comparison with source hashes,
  explicit folds, per-session losses, and synthetic recovery tests.
- Shortened the project entry points and separated current claims from historical
  records. Preserved all prior implementations, runs, and frozen interfaces.

## September 6 progress

The complete source reconciliation recovered 83 animals in 9 labs and corrected
255 misclassified omissions in a separate candidate dataset. Subject-disjoint
folds preserve the exploratory evidence-history gain. The metadata inventory
also excludes legacy-reference animals (90 known animals total).

A 60-animal cohort and an executable internal prospective protocol are frozen;
see [source reconciliation](SOURCE_RECONCILIATION.md) and
[replication protocol](REPLICATION_PROTOCOL.md). The freeze is hash-bound and
local, not an external preregistration or a committed-protocol claim.

## Completed replication — September 11

The original internal test passed without changing the frozen rules: 45/60
animals eligible, 32/45 improved, equal-animal gain 0.001598 nats/trial,
95% subject-bootstrap interval [0.000821, 0.002437]. See
[the result and limitations](REPLICATION_RESULT.md). The frozen plan was local
and uncommitted; there is no claim that the earlier desired version-control or
external-preregistration milestone had been met.

This closes the first source-verification and prospective prediction milestone.
It does not validate the adaptive controller or establish a novel mechanism.

## Development follow-up — September 12

A fixed eight-model comparison now tests evidence history against five-trial
outcome history and two simple leaky correct-side estimates. All models use the
same 84,674 eligible trials from 83 development animals, with animal-disjoint
folds and leave-one-lab-out sensitivity. The combined-control evidence gain is
0.000445 nats/trial; longer history accounts for much more predictive improvement.
See [the competing-history report](COMPETING_HISTORY.md). This is development
evidence; the completed reserved cohort remains excluded and unchanged.

The matched choice-only architecture follow-up yields reduced/full mean NLL
0.445847/0.446538 with no resolved superiority. The extra controller remains a
hypothesis, and both models remain comparators. See
[the matched result](MATCHED_CHOICE_CONTROL.md).

Synthetic positive controls recover lag-only and lag-plus-evidence generators,
but a complete model-recovery confusion matrix remains future work. The learning
alternatives are deliberately simple and use analyst block controls, so they do
not exhaust animal inference explanations. Do not compare these absolute scores
with architecture scores from different cohorts and information access.

## Next milestone: distinguish explanations

1. Compare the result with prior evidence-dependent history literature before
   asserting novelty. Identify the exact incremental claim worth testing.
2. On development animals, compare the current model with longer outcome
   histories, learning/inference models, and individual variation. Use equivalent
   information access; block identity remains an analyst control, not an oracle
   supplied to an autonomous agent.
3. Check animal and lab sensitivity. The completed test has 21/45 animals from
   one lab; subject-level intervals do not imply cross-lab generalization.
4. Extend model-recovery checks to competing explanations. Add RT distributions
   only when comparing models that actually predict RT.
5. Freeze a new test and reserve another animal cohort before confirmation.
   These 45 scored animals cannot be reused as an untouched test set after tuning.

Keep the completed score, source hashes, exclusions, and frozen coefficients.
Any follow-up using its outcomes is exploratory and must be labeled accordingly.

## Remaining discriminator after stronger history controls

Before a new confirmation cohort, test whether recent choices are estimating
persistent animal/session side preferences or slow behavioral state. Any
held-out animal's bias/state must be estimated causally from preceding trials
or a predeclared prefix, never by fitting its intercept to the full scored
session. Compare added lags on identical subsequent trials. The completed [causal preference diagnostic](BIAS_HISTORY.md) now adds a
prefix-only session intercept and slow residual filters, but not hierarchical
animal or explicit switching-state models. History/evidence gains survive these
controls, while synthetic counterexamples show that lag gains can arise from
drift alone. Therefore the next gate is published generative drift/switching vs
history model recovery and realistic power simulation, before reserving a new
confirmation cohort. [Focused literature comparison](NOVELTY_REVIEW.md).

## Decision rule

Advance a richer model only if its added structure improves the frozen primary
prediction measure on fresh subjects, the subject-level uncertainty supports
that improvement, and the relevant behavioral predictions survive appropriate
controls. Predefine a practically meaningful improvement on development data;
do not choose it after the test result.

If simple models perform equally well, report that and simplify. If the apparent
history effect disappears with stronger controls, report the confound. If it
replicates, test a specific predicted task manipulation. Any of these can be a
useful result; none is guaranteed to be novel or publication-ready.

## Deferred work and compatibility

Pause global scalar sweeps, new control circuits, and DMS expansion. PRL remains
an available simulation stress test, not animal validation. Resume transfer only
after the perceptual-task measurement and prediction gates are met; DMS then
starts with metrics and a memoryless baseline.

Keep existing CLI arguments, schema keys, paths, and defaults stable. Do not
remove historical scripts merely because they are no longer the active path.
The historical experiment record belongs in FINDINGS.md and existing artifacts;
current claims belong in README.md and the dated audit report.

## Scientific basis

The [IBL task paper](https://elifesciences.org/articles/63711) models sensory,
rewarded/unrewarded choice history, and block contributions. The reset adopts
that separation while testing an additional evidence interaction; it is not an
exact reproduction of the paper's fitting procedure.
[Wilson and Collins](https://elifesciences.org/articles/49547) motivate predictive
comparison and recovery checks. These methods support assessment of competing
explanations; they cannot establish anatomical necessity from behavioral fits.
