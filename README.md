# AnimalTaskSim

AnimalTaskSim benchmarks AI agents on classic animal decision-making tasks using task-faithful environments, public reference data, and a schema-locked evaluation stack. The project focuses on matching animal **behavioral fingerprints**—psychometric, chronometric, history, and lapse patterns—rather than raw reward.

---

## Recent Updates

### October 12, 2025 - Hybrid DDM+LSTM Agent Achieves Animal-like Chronometric Slope

After a series of targeted experiments, the hybrid DDM+LSTM agent now successfully replicates the negative chronometric slope observed in macaques. This was achieved by implementing a curriculum learning strategy that prioritizes the Wiener First Passage Time (WFPT) likelihood loss.

**Quantitative Results (`runs/hybrid_wfpt_curriculum/`):**

- **Chronometric slope:** -767 ms/unit (macaque reference: -645 ms/unit)
- **Psychometric slope:** 7.33 (macaque: 17.56)
- **Bias:** +0.001 (macaque: ≈0)
- **History effects:** All near chance, consistent with reference data.

**Technical approach:**

1. **Curriculum Learning**: A two-phase curriculum was implemented. The first phase focuses exclusively on the WFPT loss to establish the core chronometric relationship. The second phase introduces the other behavioral losses.
2. **WFPT Likelihood Loss**: This statistically principled loss function provides a much more stable and direct training signal for the DDM's parameters than the previously used Mean-Squared Error on reaction times.
3. **Non-Decision Time Supervision**: A small supervision loss was added to keep the non-decision time in a plausible range.

**Limitations:** The agent's reaction times are still globally slower than the macaques', and the psychometric slope is shallower. However, the fundamental mechanism of evidence-dependent timing has been successfully captured. See [`FINDINGS.md`](FINDINGS.md) for a complete analysis.

---

**Current scope (v0.1):**

- IBL-style mouse visual 2AFC task
- Macaque random-dot motion discrimination task
- Baseline agents: Sticky-Q, Bayesian observer, PPO
- **NEW: Hybrid DDM+LSTM agent** with stochastic evidence accumulation
- Metrics, reports, and `.ndjson` logs that align agents with rodent/primate data

Read the full benchmark recap in [`FINDINGS.md`](FINDINGS.md). Dashboards are stored under `runs/` for interactive inspection.

---

## Project Overview

- Provide reproducible Gymnasium environments that mirror lab protocols and timing.
- Train seeded baseline agents and log one JSON object per trial using the frozen schema.
- Run evaluation scripts that score fingerprints against shared reference datasets and render HTML reports.
- Design code paths so future PRL and DMS tasks drop in without breaking interfaces.

---

## Quickstart

```bash
# Install (editable + dev extras)
pip install -e ".[dev]"

# Train a baseline agent (writes runs/<name>/)
python scripts/train_agent.py --env ibl_2afc --agent sticky_q --steps 2000 --out runs/ibl_stickyq

# Train the hybrid DDM+LSTM agent (uses macaque reference data)
python scripts/train_hybrid.py --output_dir runs/rdm_hybrid --epochs 5 --episodes 10

# Evaluate fingerprints and generate a report
python scripts/evaluate_agent.py --run runs/ibl_stickyq
python scripts/make_report.py --run runs/ibl_stickyq
```

Each command respects deterministic seeding, persists `config.json`, and emits schema-validated `.ndjson` logs.

---

## Repository Layout

```text
animal-task-sim/
├─ envs/                # Gymnasium tasks + timing utilities
├─ agents/              # Sticky-Q, Bayesian observer, PPO, hybrid DDM agents
├─ eval/                # Metrics, schema validator, HTML report tooling
├─ scripts/             # Train / evaluate / report CLIs (frozen interfaces)
├─ data/                # Reference animal logs and helpers
├─ tests/               # Env/agent/metric + schema unit tests
└─ runs/                # Generated configs, logs, metrics, dashboards
```

---

## Task Snapshots

### Mouse 2AFC (IBL)

- Discrete (`left`, `right`, `no-op`) actions; contrast-driven observations in [-1, 1].
- Block priors and lapse regimes match the reference dataset; priors hidden by default.
- Sessions run for fixed trial counts and log per-phase timing.

### Macaque RDM

- Motion coherence observations with optional go-cue phases.
- Actions: `left`, `right`, `hold`, with optional per-step costs.
- Supports collapsing bounds and chronometric metrics for RT alignment.

---

## Evaluation Stack

- `scripts/train_agent.py` seeds Python/NumPy/Torch and saves configs alongside logs.
- `scripts/evaluate_agent.py` computes psychometric, chronometric, history, and bias metrics, writing `metrics.json`.
- `scripts/make_report.py` renders HTML reports that juxtapose agent runs with reference curves.
- `eval/schema_validator.py` guards the `.ndjson` contract; `tests/test_schema.py` keeps regressions from landing.

---

## Guiding Principles

- Fidelity over flash: copy lab timing, priors, and response rules exactly.
- Fingerprints over reward: success = matching bias, RT, history, lapse statistics.
- Reproducibility: deterministic seeds, saved configs, and schema-validated logs.
- Separation of concerns: environments, agents, metrics, and scripts remain decoupled.

---

## Roadmap Preview (v0.2)

- **Probabilistic Reversal Learning (PRL):** bias-block reversals with perseveration metrics delivered through the same logging schema.
- **Delayed Match-to-Sample (DMS):** delay-dependent accuracy and RT metrics sharing evaluation infrastructure.

---

## License

Code is released under MIT; datasets retain their original licenses.
