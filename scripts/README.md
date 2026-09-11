# Script guide

## Active research workflow

```bash
python scripts/audit_ibl_history.py
```

This is the September 2026 exploratory IBL history comparison. See
[methods/results](../docs/HISTORY_AUDIT.md) and [next milestone](../docs/RESEARCH_PLAN.md).
It validates the reference with the shared loader and writes separate analysis
artifacts. The data were already inspected; session folds are not unseen animals.

## Stable infrastructure

| Purpose | Entry points |
|---|---|
| Train comparison agents | `train_agent.py`, `train_r_ddm.py` |
| Train experimental mechanisms | `train_hybrid_curriculum.py`, `train_adaptive_control.py` |
| Evaluate and report | `evaluate_agent.py`, `make_report.py`, `make_dashboard.py` |
| Inspect reference | `compute_reference_targets.py` |
| Acquire reference | `fetch_ibl_reference.py`, `ibl_to_ndjson.py`, `roitman_csv_to_ndjson.py` |
| Inspect existing experiments | `compare_runs.py`, `query_registry.py`, `scan_runs.py` |

Existing names, flags, and default output paths remain stable. `max_sessions`
now counts sessions rather than training chunks in the shared Hybrid/adaptive
loader, so historical commands may train longer and do not reproduce old runs.
The `persistence_only` default is retained for compatibility; its positive retry
signature is not validated against the adopted mouse reference.

`fetch_ibl_reference.py` requires the optional ONE-api client and writes an
expanded candidate file without replacing the adopted reference. The adopted
reference already contains 120 sessions. Verify raw choice/no-go semantics and
recover subject/lab metadata before using a new pull for confirmatory research.

## Historical experiments

The sweep, calibration, injection, co-evolution, five-seed, PRL transfer, and
arbitration scripts remain in place so recorded experiments and imports remain
reproducible. They are not the recommended next workflow. Their scientific
context is in [FINDINGS.md](../FINDINGS.md); old shell wrappers are under
`docs/archive/commands/`.

No new scalar sweep or DMS architecture is planned before the replication gate.
New analyses should share reusable evaluation functions, save configs and source
hashes, and distinguish exploratory results from independent confirmation.

## Source verification and prospective replication

- `fetch_ibl_audit_sources.py`: retrieve the exact adopted sessions, with identities
  and hashed raw ALF arrays. Requires the optional ONE-api environment.
- `reconcile_ibl_reference.py`: independently check source fields, write a separate
  omission-corrected candidate, and run subject-disjoint exploratory comparisons.
- `prepare_ibl_replication.py`: metadata-only selection excluding animals from all
  local IBL reference files, including legacy references.
- `run_ibl_replication.py`: explicit freeze/download/score stages, sealed model
  coefficients, fixed eligibility, subject-level bootstrap, and a single saved score.

See [source audit](../docs/SOURCE_RECONCILIATION.md) and
[frozen protocol](../docs/REPLICATION_PROTOCOL.md). The ONE environment needs
NumPy, SciPy, pandas, Pydantic, tyro, and matplotlib because evaluation package
imports include reporting; it does not need PyTorch. Use `PYTHONPATH=.` from the
repository root when it is not installed into that environment.

## Repaired architecture validation

The full architecture and historical limitations are described in
[the repair record](../docs/ARCHITECTURE_REPAIR.md). New training uses chronological
session memory and calibrated Monte Carlo DDM expectations. Old outputs must not
be treated as repaired results.

```bash
python3 -m scripts.calibrate_ddm_surrogate --output /tmp/ddm-calibration.json
python3 -m scripts.checkpoint_ablation --source-run runs/YOUR_CHECKPOINT --output-root runs/NEW_ABLATION --reference-log runs/ibl_source_reconciled/reference.ndjson
```

The checkpoint command performs no optimization, applies four runtime profiles
to identical weights with trial-aligned randomness, and requires a fresh output
directory. Retrained validation suites now refuse stale or unverifiable results.

## Exploratory animal-prediction gate

`python3 -m scripts.validate_architecture_prediction` freezes a subject-disjoint
plan before fitting, compares simple history prediction with reduced/full
recurrent controllers, and writes per-trial and per-animal scores. Use a fresh
`--output` directory. The first recorded run failed its superiority gate; see
[the validation report](../docs/ARCHITECTURE_VALIDATION.md). Repeating or changing
this development experiment does not create a new untouched test cohort.
