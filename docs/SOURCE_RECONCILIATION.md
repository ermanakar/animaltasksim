# IBL source reconciliation — 6 September 2026

## Verified result

All **120 adopted sessions / 86,648 trials** were retrieved again from public
IBL ALF trial arrays through ONE. Session identities resolve to **83 mice in
9 laboratories**. No session was silently removed from the source audit.

The adopted reference contains **255 misclassified omissions across 49 sessions**.
Raw `choice=0` marks a no-go, but `feedbackType=-1` also applies to no-go trials.
The former importer inferred an opposite-side choice from negative feedback
on nonzero contrast. It correctly retained only 114 of 369 actual omissions.
The [official extractor source](https://docs.internationalbrainlab.org/_modules/ibllib/io/extractors/training_trials.html)
confirms both conventions.

Independent reconciliation used ALF wheel-choice coding (-1 selects the
right-side stimulus; +1 selects left), and cross-checked committed choices
against stimulus side and feedback. All adopted trial indices, contrasts,
priors, correctness, binary reward, logged predecessor fields, and response-time
values matched the retrieved source, apart from the identified omission choices.
This is verification against published ALF arrays, not re-extraction from raw
Bpod events or wheel traces.

The candidate corrects all omission choices, next-trial history, and omission
RTs (null rather than the timeout). The adopted files and old result snapshots
remain untouched. Source `.npz` files, retrieval timestamps, identities, and
SHA-256 values are retained in `runs/ibl_source_audit/`.

## Corrected, subject-disjoint exploratory analysis

`runs/ibl_source_reconciled/reference.ndjson` contains the separate candidate.
Its 86,648 records pass the shared schema. The history analysis has 85,894
eligible transitions. Its raw failure retry gap is **−0.08532**, so correcting
the omission problem does not reverse the initial observation.

All sessions of a mouse are assigned to the same fold. With five folds:

| Comparison | Trial-weighted NLL gain | Equal-mouse NLL gain | Mice improved |
|---|---:|---:|---:|
| Outcome history + nuisance terms over stimulus/block | 0.023610 | 0.023305 | 72/83 |
| Evidence-dependent history over outcome history | 0.001141 | 0.001184 | 58/83 |

Gains are in nats/trial. The evidence increment is positive in all five folds,
but one is small (0.000043). The weak-failure interaction remains negative in
all training folds (approximately −0.516 to −0.416). The old reference had
already been inspected, so this remains exploratory despite subject separation.

## Prospective cohort preparation

The metadata inventory includes the legacy ten-session and single-session
reference files too: **131 known sessions representing 90 mice** in total.
Excluding only the adopted reference would have missed seven previously used
animals. From 5,217 public matching-session metadata records, we selected 60
other animals and their latest exact standard biasedChoiceWorld session.
No optogenetic/named task variants or outcome-based selection were allowed.

The metadata sampling frame, excluded identities, selected IDs, source hashes,
and caveats are saved in `runs/ibl_replication_plan/`. The frozen test is defined
in [REPLICATION_PROTOCOL.md](REPLICATION_PROTOCOL.md). Its results are reported
separately; source reconciliation alone is not prospective replication.

## Code and reproduction

```bash
# ONE-api is an optional acquisition dependency in a separate environment.
PYTHONPATH=. /tmp/animaltasksim-one-audit/bin/python scripts/fetch_ibl_audit_sources.py
OPENBLAS_NUM_THREADS=1 python scripts/reconcile_ibl_reference.py
PYTHONPATH=. /tmp/animaltasksim-one-audit/bin/python scripts/prepare_ibl_replication.py
```

The temporary environment path is specific to this run; on another computer,
create a dedicated environment with ONE-api, tyro, and Pydantic and set the
repository root on PYTHONPATH. Acquisition does not require the model-training
PyTorch environment. The existing runtime dependencies remain unchanged.

The reconciler fails on unexplained changes in source signals or committed
choices. Its tests deliberately inject errors in choice, contrast, prior, and
RT. The importer now detects no-go before inferring side, rejects unknown
choice codes, preserves original indices, and clears history after exclusions.
Existing flags/schema types/paths remain stable; future imported values are
corrected and should not be expected to reproduce the legacy erroneous data.

No biological necessity or novelty follows from the source reconciliation.
The scientific question remains whether the small predictive increment survives
an independently selected cohort with frozen methods.
