# Internal prospective IBL replication protocol — 6 September 2026

## Timing and scope

This protocol is fixed after inspecting the development reference and before
loading the 60 reserved animals' ALF trial arrays in this run. The executable
freeze records its SHA-256, fitted coefficient hashes, source hashes, settings,
and time. It is an internal prospective test, not external preregistration or
a claim that the protocol was committed before analysis.

The 60 candidates are absent from all three inventoried local IBL references.
Earlier unrecorded downloads or inspections cannot be excluded. We will describe
them as a new-to-reference cohort, rather than claim a provably pristine test set.

## Hypothesis and primary test

Prior stimulus strength modulates the predictive contribution of outcome-linked
choice history beyond current stimulus/block cells and ordinary outcome history.
The primary comparison is **evidence_history vs outcome_history**, as defined
in `eval/history_audit.py`. The added model includes both rewarded and unrewarded
choice × weak-prior-evidence interactions. This is a joint two-term prediction
test, not evidence that either term independently causes behavior.

Primary score: mean across eligible test animals of (outcome_history negative
log likelihood minus evidence_history negative log likelihood), in nats/trial.
Each animal contributes equally. Both models predict identical eligible trials.

Success requires all of:

- At least 40 eligible animals from the reserved cohort.
- Mean improvement at least 0.0005 nats/trial.
- Lower bound of the 95% subject-bootstrap interval strictly above zero.

The 0.0005 screen is an internal practical criterion, about half the exploratory
benefit, not an established biological threshold. Use 10,000 percentile bootstrap
resamples of test subjects, seed 20260906, conditional on the fixed models.
No fit, tuning, or model selection uses reserved-cohort outcomes.

## Models and training

Use the three existing logistic models without changing feature definitions.
Fit once to the reconciled development reference (120 sessions, 83 animals).
Use fixed L2 = 1 on summed binomial loss; leave the intercept unpenalized.
Use weak evidence |previous contrast| ≤ 0.125. Freeze coefficient arrays and
feature order before acquisition. The stimulus/block-only model is descriptive
context; it is not an additional primary hypothesis test.

## Sampling frame and exclusions

The metadata frame contains 5,217 sessions returned by the public
`biasedChoiceWorld` query. Keep only exact standard `_iblrig_tasks_biasedChoiceWorld`
versioned protocols, excluding optogenetic and other named variants. Exclude
all 90 animals represented by the 131 sessions across the three local reference
files. Hash-order remaining subject names with seed 20260906 and take 60; use
each subject's latest standard session, breaking date ties by EID. The full
metadata frame and chosen IDs are saved. No performance fields are used.

Prespecified eligibility:

- At least 150 retained trials and 20 full-contrast trials in a session.
- Accuracy at full contrast ≥0.85, counting omissions as failures.
- Known ALF choices (-1 selects right, +1 selects left, 0 omits), feedback,
  source alignment, stimulus side, and standard block priors.
- Retain only |contrast| in {0, 0.0625, 0.125, 0.25, 1}. Preserve original indices
  and clear history at excluded rows. Unknown or inconsistent source coding
  excludes a session and is reported.
- Build predecessors before analysis exclusions; require adjacent committed
  current and previous trials. Omission RT is null. RT is not a primary outcome.

No replacement subjects after QC or disappointing results. Download failures
may be retried for the same fixed EIDs. Incomplete acquisition prevents scoring;
fewer than 40 eligible subjects yields an insufficient-sample result, not a pass.

## Reporting and decision

Report every candidate's QC outcome, sample sizes, per-animal scores, the primary
mean/interval, and criterion outcome, including a negative or inconclusive result.
Do not change models or thresholds after scoring. Any subsequent analysis is
explicitly exploratory and cannot replace this result.

Passing supports generalization of a narrow behavioral predictor to this selected
cohort. It does not validate the adaptive agent, subjective uncertainty, neural
anatomy, RT dynamics, mechanistic necessity, lab-wide generalization, or novelty.
A failure is a result to report and investigate, not a reason to resume sweeps.
