# IBL history audit — 5 September 2026

> Superseded for source handling and independence by the
> [September 6 reconciliation](SOURCE_RECONCILIATION.md). The numbers below are
> the original exploratory snapshot; they are retained without retrospective rewriting.


## Finding and interpretation

The adopted reference does not support using a positive raw weak-minus-strong
failure retry gap as the animal target. The observed gap is **−0.0830**:
P(retry | weak failure) = 0.5457 (14,524 transitions), versus 0.6287 for strong
failures (2,268 transitions). This is a conditional association, not a causal
response to uncertainty.

With a fixed weak threshold of |prior contrast| ≤ 0.125 and at least ten
failures in each bin, 84 sessions qualify. Their median gap is −0.0607,
IQR [−0.1174, +0.0052]; 27.4% have a positive gap. Sessions are not necessarily
independent animals. No population significance claim is made.

A controlled predictive comparison provides a second observation:

| Model | Session-held-out log loss (nats/trial) | Gain over preceding model | Sessions improved |
|---|---:|---:|---:|
| Current stimulus × block | 0.417684 | — | — |
| Outcome history + nuisance terms | 0.393875 | 0.023809 | 101/120 |
| Evidence-dependent history | 0.392698 | 0.001177 | 78/120 |

Lower loss is better. The final increment is about 0.30% of the preceding
model's loss. It is positive in all five held-out folds (0.000564–0.001832
nats/trial). Equal-session averaging gives 0.001188 nats/trial. These are
exploratory effect sizes, not independent-fold significance tests.

The fitted `unrewarded_choice_weak` coefficient is negative in all training
folds (−0.475 to −0.405). The direction remains opposite to the controller's
intended weak-failure persistence effect after the specified controls. This
coefficient is a signed-choice interaction on the logit scale, not the raw
retry probability gap. It does not establish subjective confidence or identify
a neural circuit. The improvement includes both rewarded and unrewarded
interactions; it cannot be attributed solely to the failure term.

**Research lead:** evidence-dependent outcome history is worth replicating,
but its direction and explanatory value must be established from animals
before choosing an adaptive mechanism. These results do not validate the
existing controller, establish novelty, or constitute a completed publication.

## Fixed exploratory method

- Source: adopted IBL reference, 86,648 trials / 120 sessions. The existing
  Pydantic loader validates each record. Its manifest reports no dropped rows.
- Build predecessors within sorted sessions before exclusions; require adjacent
  logged indices and committed current/previous choices. Retain 86,312 trials.
- Use exact protocol contrast categories, tolerating numerical roundoff only.
- The stimulus/block model has an intercept and 26 categorical cell indicators
  (nine signed contrasts × three block priors, with one reference cell). This
  allows different choice curves in different blocks without a shared slope.
- Outcome history adds previous success, signed rewarded/unrewarded choice,
  categorical previous strength, and strength × previous-success nuisance
  terms. Thus its total gain is not attributable to choice history alone.
- The final model adds signed rewarded/unrewarded choice × previous weak
  evidence. All three models use precisely the same eligible trials.
- Ridge logistic fits minimize summed binomial negative log likelihood plus
  0.5 × ||beta||², excluding the intercept penalty. L2 = 1 is fixed, not selected
  from held-out results. Optimizer failure raises an error.
- Seed 20260905 assigns entire sessions to five disjoint folds. Each row is
  predicted once by a model trained without its session. Fold lists and fitted
  coefficients are saved. Held-out animal histories are observed conditioning
  variables: this is predictive analysis, not free-running simulation.
- Unit tests cover synthetic interaction recovery and a null interaction,
  complete/disjoint deterministic folds, and omission/session/gap exclusions.
  These are sanity checks, not a full model-identifiability study.

## Reproduce and inspect

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_ibl_history.py
MPLBACKEND=Agg pytest tests/test_history_audit.py tests/test_metrics.py
```

`runs/ibl_history_refresh/config.json` records reference/manifest/code SHA-256,
analysis settings, and package versions. `metrics.json` contains full results,
fold membership, per-session scores/counts, and coefficients. The tracked
[snapshot](results/history_audit_2026-09-05.json) includes both files. Re-running
with the same code, source, settings, and libraries should reproduce the
analysis; small numerical differences across library/platform versions remain
possible. Never silently replace the historical sweep metrics with this audit.

## What remains unresolved

1. Animal/lab IDs are absent from the local manifest. Session-held-out prediction
   is not subject-held-out prediction, and bootstrap over sessions would not
   fix that. Obtain metadata and a fresh cohort before confirmation.
2. The reference and raw gap were inspected before this comparison was designed.
   There is no pristine test set here and no preregistration claim.
3. Raw-source choice/no-go conversion needs independent verification. The
   importer derives nonzero-contrast choices from feedback; feedback-negative
   omissions could be misclassified as opposite-side choices. This audit validates
   the existing derived file, not a fresh reconciliation against ONE source arrays.
4. One-trial history and observed block controls do not exhaust explanations.
   Longer histories, individual differences, latent learning states, and stimulus
   sequence effects remain competing explanations. Block identity is available
   to the analyst, not as an agent observation.
5. Existing simulated IBL blocks cycle fixed lengths and include repeated neutral
   blocks; published IBL blocks are variable after an initial neutral block.
   Simulator RT starts at the response phase; reference RT is stimulus-onset to
   response. These must be reconciled before claiming matched task/RT dynamics.
6. Omissions are now excluded from directional fits and stay/switch metrics.
   Existing saved metrics were not regenerated. The legacy history coefficients
   remain descriptive/unadjusted; use this separate analysis for controlled
   prediction. Legacy RT ceiling detection and heuristic quality targets also
   remain unsuitable as scientific acceptance gates.

The next steps and decision rules are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).
The [IBL methods](https://elifesciences.org/articles/63711) motivate controlling
sensory evidence, outcomes, and blocks. The recovery/comparison approach follows
[Wilson and Collins](https://elifesciences.org/articles/49547), while the exact
models and limitations above are specific to this repository.
