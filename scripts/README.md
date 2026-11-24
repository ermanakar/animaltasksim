# Script guide

## Active research workflow

The current direction is within-animal slow-state modelling on the deep
development reference; see the [research plan](../docs/RESEARCH_PLAN.md).

| Purpose | Entry point |
|---|---|
| Acquire every standard session of the development animals | `fetch_ibl_deep_reference.py` (stages `inventory`, `plan`, `download`, `verify`; needs ONE-api) |
| Convert the deep reference to schema trials and run its gate | `convert_ibl_deep_reference.py` (stages `convert`, `gate`) |
| First exploratory look at the deep reference | `explore_deep_reference.py` |
| Sensitivity to optogenetic and RPE sessions | `exclude_variant_sensitivity.py` (stages `prepare`, `generative`) |
| Evidence gain split by previous outcome | `split_evidence_by_outcome.py` |
| Recovery of within-animal models on real schedules | `recover_within_animal.py` (stages `plan`, `run`, `summarize`) |
| Revision 1 of that recovery study | `recheck_within_animal_recovery.py` (same stages) |
| Figures for the two reports above | `render_state_direction_figures.py` (reads tracked snapshots only) |

Reports: [protocol](../docs/DEEP_REFERENCE_PROTOCOL.md) ·
[conversion](../docs/DEEP_CONVERSION_PROTOCOL.md) ·
[recovery](../docs/WITHIN_ANIMAL_RECOVERY_PROTOCOL.md) ·
[first look](../docs/DEEP_EXPLORATION.md) ·
[sensitivity](../docs/VARIANT_SENSITIVITY.md) ·
[outcome split](../docs/OUTCOME_SPLIT.md).

```bash
python scripts/audit_ibl_history.py
```

This is the September 2026 exploratory IBL history comparison. See
[methods/results](../docs/HISTORY_AUDIT.md).
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

No new scalar sweep or DMS architecture is planned under the current plan.
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

### Stronger history controls (development analysis)

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m scripts.compare_history_explanations`
runs eight fixed nested history models on the source-reconciled development
cohort, with common contiguous histories, animal folds and leave-one-lab-out
sensitivity. The runner freezes its plan before fitting, excludes the reserved
cohort, and requires a fresh output directory. See
[the protocol and result](../docs/COMPETING_HISTORY.md).

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m scripts.choice_control_comparison`
completes the matched choice-only reduced-model fit against saved full-model
predictions. Seeds, checkpoints and prediction paths are fixed; prior artifacts
are hash-checked and output must be fresh. See
[the matched result](../docs/MATCHED_CHOICE_CONTROL.md).

### Causal preference controls and synthetic recovery

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m scripts.diagnose_bias_history`
uses a new frozen plan, excludes prefix trials from all scores, and runs fixed
bias/drift/history synthetic diagnostics after the development comparison.
`MPLBACKEND=Agg python3 -m scripts.render_bias_history_figure` renders the saved
snapshot. See [results and identification limits](../docs/BIAS_HISTORY.md).

## Generative history versus latent preference (September 2026)

This development-only comparison has its own [protocol](../docs/GENERATIVE_COMPARISON_PROTOCOL.md).
It preserves the frozen replication CLI and data. Existing verified source-reconciled
inputs and the September 12 comparison plan must be present; it downloads no cohorts.

```bash
pip install -e '.[dev,research]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.compare_generative_history
# Resume completed recovery cells only with identical source, inputs and settings:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.compare_generative_history --resume
python -m scripts.verify_generative_history
OPENBLAS_NUM_THREADS=1 python -m scripts.check_drift_resolution
# Only after the main run is complete:
MPLBACKEND=Agg python -m scripts.render_generative_comparison
```

The full recovery batch is a research run, not a short demo: 360 development-sized
simulations and complete refits. `--repetitions 1 --output runs/<new-name>` is a
numerical smoke test, not a qualified power result. Keep BLAS threads at one when
using multiple workers. Optional Numba acceleration is strongly recommended for
research runs; the module has a slower NumPy reference fallback.

`check_published_state_filter` additionally requires the author `ssm` fork at
`e9b408b79e2ab22a05ca93c3a1f78a7dae461992`, installed in an isolated research environment.
The script checks installed source against that checkout and verifies emissions
and causal filtering, **not** independent model fitting. The local checked
checkout is `runs/vendor_ssm_20260919`; all installed package versions are saved
in the experiment's `environment.json`. This optional legacy dependency is not
added to normal application or CI requirements.

After installing the pinned author fork in that isolated environment, the optional
independent checks are:

```bash
OPENBLAS_NUM_THREADS=1 python -m scripts.check_published_state_filter
OPENBLAS_NUM_THREADS=1 python -m scripts.refit_author_glmhmm
```

The refit uses the author package's likelihood and autograd derivatives on our
matched design, priors and folds. It is not a reproduction of the paper's dataset
selection or full analysis. Both checks refuse to overwrite their existing plans
or require the checked upstream revision as appropriate.
