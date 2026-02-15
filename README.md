# AnimalTaskSim

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen.svg)](tests/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

AnimalTaskSim benchmarks AI agents on classic animal decision-making tasks using task-faithful environments, public reference data, and a schema-locked evaluation stack. The project focuses on matching animal **behavioral fingerprints**—psychometric, chronometric, history, and lapse patterns—rather than raw reward.

## Current Results

<p align="center">
  <img src="docs/figures/k2_dashboard.png" alt="Agent vs Macaque Behavioral Fingerprints" width="800">
</p>

The Hybrid DDM+LSTM agent trained on the macaque RDM task simultaneously reproduces:
- **Psychometric sensitivity** — sigmoidal choice curve with slope 10.7, near-zero lapses (macaque reference: 17.6)
- **Negative chronometric slope** — harder stimuli → slower RTs, slope −270 ms/unit (macaque: −645)
- **Near-zero choice bias** — p(right) = 0.496, bias = 0.002 (macaque: ≈0)
- **History effects at chance** (~0.50) — an open research problem called the [Decoupling Problem](FINDINGS.md#the-decoupling-problem)

> *Figure from `runs/decoupling_K2_window_control/`. Regenerate with `python scripts/generate_readme_figures.py`.*

---

## New to the Project?

- 📘 **[Theory & Concepts Guide](docs/THEORY_AND_CONCEPTS.md)** — Accessible introduction covering the tasks, behavioral fingerprints, and the hybrid DDM+LSTM model
- **[Findings Report](FINDINGS.md)** — Experimental results across 55+ runs, what works, what doesn't
- 💻 **[Agent Operating Guide](AGENTS.md)** — Implementation standards and contribution guidelines
- ⚡ **[Quick Start](#quickstart)** — Jump straight to running experiments

---

## Project Overview

AnimalTaskSim provides:

- **Task-faithful Gymnasium environments** that mirror lab protocols and timing for the IBL mouse 2AFC and macaque RDM tasks
- **Baseline agents** (Sticky-Q, Bayesian observer, PPO) and advanced agents (Hybrid DDM+LSTM, R-DDM) with deterministic seeding
- **Schema-validated trial logging** — one JSON object per trial using a frozen `.ndjson` schema
- **Evaluation scripts** that score behavioral fingerprints against reference datasets and render HTML dashboards
- **Experiment registry** for tracking and comparing runs

The code is designed so future tasks (PRL, DMS) can be added without breaking existing interfaces.

---

## Reference Data

AnimalTaskSim benchmarks against two canonical datasets from decision neuroscience:

### IBL Mouse 2AFC

**Source:** International Brain Laboratory (IBL) standardized protocol
**Task:** Two-alternative forced choice with varying visual contrasts
**Species:** Laboratory mice across multiple institutions
**Citation:** [International Brain Laboratory (2021). *Neuron*](https://doi.org/10.1016/j.neuron.2021.04.001)

The IBL dataset provides reproducible measurements of mouse decision-making behavior with controlled contrast levels, block structure, and lapse regimes. `data/ibl/reference.ndjson` bundles 10 public IBL sessions (8,406 trials); the legacy single-session log is available as `data/ibl/reference_single_session.ndjson`.

### Macaque Random-Dot Motion

**Source:** Shadlen lab perceptual decision-making studies
**Task:** Random-dot motion (RDM) direction discrimination
**Species:** Rhesus macaques
**Citations:**

- [Britten et al. (1992). *Journal of Neuroscience*](https://doi.org/10.1523/JNEUROSCI.12-12-04740.1992)
- [Palmer, Huk & Shadlen (2005). *Journal of Vision*](https://doi.org/10.1167/5.5.1)

The macaque RDM data captures the relationship between motion coherence, reaction times, and accuracy. These studies established the neural basis of evidence accumulation in area MT and inspired the drift-diffusion model framework used in our hybrid agent.

---

## Quickstart

### Installation

```bash
pip install -e ".[dev]"
```

### Interactive Workflow (Recommended) ⭐

```bash
python scripts/run_experiment.py
```

This guides you through selecting a task, choosing an agent, configuring parameters, training, evaluation, dashboard generation, and registry update.

### Manual Workflow

```bash
# 1. Train an agent
python scripts/train_agent.py --env ibl_2afc --agent ppo --episodes 5 --seed 42 --out runs/my_experiment

# 2. Evaluate behavioral metrics
python scripts/evaluate_agent.py --run runs/my_experiment

# 3. Generate interactive dashboard
python scripts/make_dashboard.py \
  --opts.agent-log runs/my_experiment/trials.ndjson \
  --opts.reference-log data/ibl/reference.ndjson \
  --opts.output runs/my_experiment/dashboard.html

# 4. Update experiment registry
python scripts/scan_runs.py --overwrite

# 5. Query results
python scripts/query_registry.py show --run-id my_experiment
```

### Experiment Registry

```bash
python scripts/query_registry.py list                                       # All experiments
python scripts/query_registry.py list --task rdm_macaque --agent hybrid_ddm_lstm  # Filtered
python scripts/query_registry.py show --run-id my_experiment                # Detail view
python scripts/query_registry.py export --output experiments.csv            # CSV export
```

The registry (`runs/registry.json`) tracks task, agent, seed, training parameters, behavioral metrics, file paths, status, and timestamps.

---

## Repository Layout

```text
animal-task-sim/
├─ envs/                # Gymnasium tasks + timing utilities
├─ agents/              # Sticky-Q, Bayesian observer, PPO, hybrid DDM, R-DDM agents
├─ animaltasksim/       # Core utilities (config, logging, seeding, registry)
├─ eval/                # Metrics, schema validator, HTML report tooling
├─ scripts/             # Train / evaluate / report CLIs (frozen interfaces)
├─ data/                # Reference animal logs and helpers
├─ tests/               # Env/agent/metric + schema unit tests (80+)
├─ docs/                # Documentation and guides
└─ runs/                # Generated configs, logs, metrics, dashboards
   ├─ archive/          # Archived experimental runs
   └─ registry.json     # Experiment database
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
- `scripts/make_dashboard.py` renders interactive HTML dashboards that juxtapose agent and reference curves.
- `scripts/compare_runs.py` scans all runs and produces a color-coded multi-run leaderboard.
- `eval/schema_validator.py` guards the `.ndjson` contract; `tests/test_schema.py` keeps regressions from landing.

---

## Guiding Principles

- **Fidelity over flash:** copy lab timing, priors, and response rules exactly.
- **Fingerprints over reward:** success = matching bias, RT, history, lapse statistics.
- **Reproducibility:** deterministic seeds, saved configs, and schema-validated logs.
- **Separation of concerns:** environments, agents, metrics, and scripts remain decoupled.

---

## Reproducibility

Every experiment is fully reproducible via fixed seeds, saved `config.json`, schema-validated logs, and registry tracking. To reproduce:

```bash
python scripts/query_registry.py show --run-id EXPERIMENT_NAME  # View config
python scripts/train_agent.py --seed SEED --env TASK --agent AGENT ...  # Re-run
```

---

## Recent Highlights

### February 2026 — K2 Breakthrough

The K2 experiment achieved simultaneous replication of psychometric sensitivity and negative chronometric slope. Key discoveries:

1. **Bias artifact resolution**: The reported 84% "leftward bias" was a metric artifact — `p_right_overall` counted holds as non-right. The true committed `p_right = 0.48` was balanced all along.
2. **Response window fix**: The agent's `max_commit_steps=200` exceeded the environment's 120-step response phase. After alignment, commit rate reached 100%.
3. **Results** (`runs/decoupling_K2_window_control/`): Psych slope=10.7 (lapses ≈0), chrono slope=−270 ms/unit, bias=0.002, commit rate=100%.

### February 2026 — Scientific Fixes & Infrastructure

- **WFPT normalization fix**: Both small-time/large-time series now agree to 6 decimal places; `drift=3, bound=2` integrates to exactly 1.000.
- **Per-trial history loss**: Root cause of the Decoupling Problem identified. Added `per_trial_history_loss()` with per-trial MSE supervision.
- **Ceiling-corrected chronometric slope**: `corrected_slope` excludes ceiling levels for more honest assessment.
- **Multi-run leaderboard**: `scripts/compare_runs.py` scans all runs, computes composite scores, outputs color-coded HTML.
- **80 tests** now pass (up from 40).

See [`FINDINGS.md`](FINDINGS.md) for full experimental details.

---

## Roadmap (v0.2)

### New Tasks

- **Probabilistic Reversal Learning (PRL):** bias-block reversals with perseveration metrics. Schema extensions already drafted and tested.
- **Delayed Match-to-Sample (DMS):** delay-dependent accuracy and RT metrics.

### Agent Improvements

- **Hybrid DDM+LSTM for IBL 2AFC:** Adapt the hybrid agent for mouse data, including contrast-level curriculum and mouse behavioral fingerprints.

---

## Acknowledgements

This project was developed independently outside of academia, with substantial contributions from AI coding assistants (Claude, OpenAI Codex, Google Gemini). It represents a collaboration between human domain expertise and AI implementation capability.

### Scientific Foundation

- **DDM:** Ratcliff & McKoon (2008); Ratcliff & Smith (2016); Wiecki et al. (2013); Navarro & Fuss (2009)
- **Animal data:** International Brain Laboratory (2021, *Neuron*); Britten et al. (1992); Palmer, Huk & Shadlen (2005)
- **History effects:** Urai et al. (2019, *Nature Communications*)

### Open Source

Built with Gymnasium, PyTorch, Stable-Baselines3, Pydantic, and the Scientific Python ecosystem (NumPy, SciPy, pandas, matplotlib).

---

## License

Code is released under MIT; datasets retain their original licenses.

---

## Citation

If you use AnimalTaskSim in your research, please cite:

```bibtex
@software{akar2025animaltasksim,
  author = {Akar, Erman},
  title = {AnimalTaskSim: Hybrid Drift-Diffusion × LSTM Agents Matching Animal Decision Behavior},
  year = {2025},
  url = {https://github.com/ermanakar/animaltasksim},
  version = {0.1.0}
}
```

See [`CITATION.cff`](CITATION.cff) for additional citation metadata and references.
