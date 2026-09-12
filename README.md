# AnimalTaskSim

**Test explanations of animal decisions against trial-level behavioral data.**

AnimalTaskSim provides task environments, comparison agents, schema-validated
trial logs, and a shared evaluation pipeline. The architectural research question is:

> Does separating longer-term memory from fast outcome-driven adaptation improve
> predictions of animal choices and response times, and when does it matter?

The completed evidence-history replication below establishes a behavioral effect
to explain; it does not validate the full architecture. The September architecture
repair corrects feedback access, DDM conventions, training continuity and experiment
controls. [Repair details and validation](docs/ARCHITECTURE_REPAIR.md).

We evaluate explanations by held-out prediction and behavioral fingerprints,
not reward alone. A software lesion establishes a contribution within a model;
it does not establish a necessary mechanism in an animal.

## Architecture validation

The repaired full model **did not pass** its first exploratory prediction gate.
After fitting on 12 mice and evaluating eight different development mice, it
improved choice prediction over the reduced recurrent model but lost to a simple
one-step history predictor. No clear mean-RT advantage was established. This
bounded two-epoch check is not a fresh prospective confirmation or a verdict on
all possible training settings. [Results and recorded protocol](docs/ARCHITECTURE_VALIDATION.md).

A completed six-epoch fitting diagnosis improves the picture: choice-only full
model NLL is **0.44654**, versus **0.46450** for simple history. Timing error
increases, and the adaptive controller's added value remains unestablished.
This reuses development animals and does not replace the original failed gate.
[Controlled diagnosis, curves and limitations](docs/FITTING_DIAGNOSIS.md).

## Current result

A frozen internal replication **passed on 45 eligible mice** from a cohort of
60 animals absent from the inventoried local references. Evidence-dependent
history improved prediction by **0.00160 nats/trial**, with a 95% subject-bootstrap
interval of **[0.00082, 0.00244]**; 32/45 mice improved. No test-data fitting,
threshold changes, or replacement subjects were used.

This is a small predictive effect, not proof of a neural mechanism or scientific
novelty. The cohort is uneven across labs, and the freeze was internal rather
than externally preregistered. [Full result and limitations](docs/REPLICATION_RESULT.md).

## Start here

```bash
pip install -e ".[dev]"
MPLBACKEND=Agg pytest
PYTHONPATH=. MPLBACKEND=Agg python scripts/render_ibl_replication_figure.py
```

The figure command reads the tracked result snapshot and needs no new download
or model training. The existing scored cohort is preserved against overwrite.

- [Completed replication: scores and exclusions](docs/REPLICATION_RESULT.md)
- [Frozen protocol](docs/REPLICATION_PROTOCOL.md)
- [Next research question](docs/RESEARCH_PLAN.md)
- [Source reconciliation](docs/SOURCE_RECONCILIATION.md)
- [Original exploratory audit](docs/HISTORY_AUDIT.md)
- [Historical experiments](FINDINGS.md)

## Why source verification mattered

The adopted reference's 120 sessions represent **83 mice across 9 labs**.
Reconciliation against public ALF arrays found **255 misclassified omissions**
in 49 sessions. A separate corrected candidate preserves the adopted file.
The corrected raw failure retry gap is **−0.0853**, opposite to the positive
signature targeted by the earlier adaptive controller.

Subject-disjoint development comparisons retained a small evidence-history
benefit. That motivated the frozen reserved-cohort test above; it does not
validate the earlier controller. Acquisition, reconciliation, metadata-only
cohort selection, and freeze/download/score commands are in the
[script guide](scripts/README.md).

## What is active

| Area | Status |
|---|---|
| IBL reference and history analysis | Active: validate the phenomenon and compare simple explanations |
| Trial logging, schema, metrics, tests | Shared scientific infrastructure |
| Sticky-Q, Bayes, DDM, PPO, Hybrid, R-DDM | Retained comparison implementations; no new calibration sweeps planned |
| Hybrid, adaptive control and PRL | Repaired implementation; controlled architectural comparisons are the next validation gate |
| DMS | Environment scaffold; expansion deferred until the IBL validation milestone |

Historical commands retain their names, flags, and output paths. Existing
`persistence_only` defaults remain for compatibility, **not as an endorsement
of an animal-validated mechanism**. See [script guide](scripts/README.md).

## Pipeline

```text
Agent → Environment → validated trials.ndjson → evaluator → metrics/report
Animal reference ───→ validated trials.ndjson → held-out model comparison
```

The environment owns agent trial logging. Offline analyses consume logs and
write separate analysis artifacts; they do not fabricate environment trials.

```bash
python scripts/train_agent.py --help
python scripts/evaluate_agent.py --run runs/your_run
python scripts/make_report.py --run runs/your_run
```

## Data and limits

The adopted IBL dataset has **120 sessions / 86,648 trials**. The macaque RDM
reference has 2,611 trials. PRL and DMS currently have no animal reference here.
IBL data attribution and CC-BY 4.0 terms are in [data/README.md](data/README.md);
repository code is MIT.

The environments are task abstractions. In particular, simulated IBL block
lengths and response timing differ from the published protocol. Existing
chronometric and lesion results need those qualifications. Old figures use an
older reference and are retained under `docs/figures/` as historical artifacts.

**Next decision:** compare evidence dependence with longer-history and
block-learning explanations on development animals. Preserve this completed
cohort result; reserve another cohort before further confirmation.
