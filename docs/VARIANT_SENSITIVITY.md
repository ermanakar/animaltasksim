# Sensitivity to optogenetic and RPE variant sessions — November 23, 2026

**Status: exploratory sensitivity check on previously inspected development animals.**

[Portable results](results/variant_sensitivity_v1.json)

## What was found

The deep-reference inventory showed that the adopted 120-session reference
contains **16 sessions that are not standard `biasedChoiceWorld`**: 14 with
optogenetic stimulation protocols and 2 with an altered reward (RPE) protocol,
from two labs (hoferlab 11, mainenlab 5). They hold 13,265 of 86,648 trials
(15.3%) and are the only sessions of 12 of the 83 development animals. The
original fetcher matched the protocol name loosely; the frozen replication
used an exact rule and is unaffected.

The adopted reference is not changed. This check asks whether the development
results depended on those sessions.

## Method

Keep only sessions whose protocol is exact versioned `biasedChoiceWorld`
(104 sessions, 71 animals). Rerun the fixed development analyses with unchanged
settings on the filtered data. Folds are redrawn by the same seeded rule because
the animal set changed.

## Result

Nothing changes sign and nothing moves by more than its own interval.

| Measure | All 120 sessions | Without the 16 |
|---|---:|---:|
| Raw weak-minus-strong failure retry gap | −0.0853 | −0.0913 |
| Audit: outcome history over stimulus and block (equal-animal) | 0.0233 | 0.0218 |
| Audit: evidence history over outcome history (equal-animal) | 0.00118 | 0.00132 |
| Competing: evidence over one-step outcome history | 0.00131 [0.00080, 0.00181] | 0.00139 [0.00078, 0.00200] |
| Competing: evidence over combined history (primary) | 0.00044 [0.00020, 0.00070] | 0.00051 [0.00020, 0.00081] |
| Competing: combined history over one-step history | 0.0254 | 0.0249 |
| Bias controls: evidence over bias plus long history | 0.00042 [0.00016, 0.00069] | 0.00049 [0.00016, 0.00082] |

Gains are choice NLL reductions in nats per trial with subject-bootstrap 95%
intervals where shown.

## Not complete

The generative comparison (drift and three-state switching models) was being
refitted on the filtered data and **had not finished when this was written**.
No conclusion is drawn about it. A subgroup split of the existing fits gives
history gains of 0.0138 (drift) and 0.0147 (switching) for the standard-session
animals, against 0.0146 and 0.0143 for all animals; that is a split of old
predictions, not a refit.

## Limits

- Removing sessions also removes 12 animals, so this tests both at once.
- The filtered cohort is smaller; intervals are slightly wider.
- This is the same inspected cohort. It supports robustness of the development
  results, not confirmation.

## Consequence

Future reference pulls use the exact protocol rule in
`scripts/fetch_ibl_deep_reference.py`. Whether to rebuild the adopted reference
without the variant sessions is a dated provenance decision that has not been
taken.

## Reproduction

```bash
python -m scripts.exclude_variant_sensitivity --stage prepare
python -m scripts.compare_history_explanations --reference runs/variant_sensitivity_20261123/reference.ndjson \
  --identities runs/variant_sensitivity_20261123/source_manifest.json \
  --reconciliation runs/variant_sensitivity_20261123/config.json \
  --output runs/competing_history_novariant_20261123
python -m scripts.diagnose_bias_history --previous-plan runs/competing_history_novariant_20261123/plan.json \
  --output runs/bias_history_novariant_20261123
python -m scripts.exclude_variant_sensitivity --stage generative
```
