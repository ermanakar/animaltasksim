# AnimalTaskSim

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen.svg)](tests/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

AnimalTaskSim benchmarks AI agents on classic animal decision-making tasks using task-faithful environments, public reference data, and a schema-locked evaluation stack. The project focuses on matching animal **behavioral fingerprints**—psychometric, chronometric, history, and lapse patterns—rather than raw reward.

---

## 📚 New to the Project? Start Here(")

**Not sure where to begin?** We've got you covered:

- 📘 **[Theory & Concepts Guide](docs/THEORY_AND_CONCEPTS.md)** — Start here! Accessible introduction for all backgrounds (neuroscientists, ML researchers, students, curious visitors)
  - Why replicate animal behavior?
  - How do the tasks work?
  - What are behavioral fingerprints?
  - How does the hybrid DDM+LSTM model work?
  
- **[Findings Report](FINDINGS.md)** — Experimental results, what works, what doesn't, and lessons learned
- 💻 **[Agent Operating Guide](AGENTS.md)** — Implementation standards and contribution guidelines for developers
- ⚡ **[Quick Start](#quickstart)** — Jump straight to running experiments (see below)

**TL;DR:** This project bridges neuroscience and AI by training agents that don't just win, but exhibit realistic animal-like decision-making patterns—including biases, history effects, and reaction time dynamics.

---

## Recent Updates

### October 12, 2025 - Hybrid DDM+LSTM Agent Achieves Animal-like Chronometric Slope

After a series of targeted experiments, the hybrid DDM+LSTM agent now successfully replicates the negative chronometric slope observed in macaques. This was achieved by implementing a curriculum learning strategy that prioritizes the Wiener First Passage Time (WFPT) likelihood loss.

**Quantitative Results (`runs/hybrid_wfpt_curriculum/`):**

- **Chronometric slope:** -767 ms/unit (macaque reference: -645 ms/unit)
- **Psychometric slope:** 7.33 (macaque: 17.56)
- **Bias:** +0.001 (macaque: ≈0)
- **History effects:** Near chance (~0.5), which does **not** match macaque reference (win-stay ~0.49, lose-shift ~0.52). See [The Decoupling Problem](FINDINGS.md#the-decoupling-problem).

**Technical approach:**

1. **Curriculum Learning**: A two-phase curriculum was implemented. The first phase focuses exclusively on the WFPT loss to establish the core chronometric relationship. The second phase introduces the other behavioral losses.
2. **WFPT Likelihood Loss**: This statistically principled loss function provides a much more stable and direct training signal for the DDM's parameters than the previously used Mean-Squared Error on reaction times.
3. **Non-Decision Time Supervision**: A small supervision loss was added to keep the non-decision time in a plausible range.

**Limitations:** The agent's reaction times are still globally slower than the macaques', and the psychometric slope is shallower. However, the fundamental mechanism of evidence-dependent timing has been successfully captured. See [`FINDINGS.md`](FINDINGS.md) for a complete analysis.

---

### October 14, 2025 - Time-Cost Curriculum Guardrails

We revisited the hybrid curriculum to keep WFPT loss dominant while widening the agent’s response window (`max_commit_steps = 180`). The updated run (`runs/hybrid_wfpt_curriculum_timecost/`) restores a negative chronometric slope without saturating the 1.2 s cap.

- **Chronometric slope:** −267 ms/unit (still shallower than macaque −645 ms/unit, but no longer flat)
- **RT intercept:** 883 ms (improved from 1.26 s, still slower than 760 ms target)
- **Psychometric slope:** 7.50 (agent remains conservative relative to 17.56 reference)
- **History:** Win-stay 0.22 / Lose-shift 0.47 / Sticky 0.42 — indicates the agent under-utilises recent rewards and is less perseverative than macaques.

The guardrails (longer WFPT warm-up, gentler RT penalties, higher commit window) keep the optimisation scientifically honest while we continue exploring soft RT regularisers and phase-level diagnostics. See [`runs/hybrid_wfpt_curriculum_timecost/dashboard.html`](runs/hybrid_wfpt_curriculum_timecost/dashboard.html) for the full comparison.

### October 15, 2025 - Soft RT Penalty Sweep

We introduced a soft reaction-time penalty that targets macaque mean RTs without forcing mean-squared-error. Early sweeps (`runs/hybrid_wfpt_curriculum_timecost_soft_rt/`) keep the WFPT warm-up intact but reveal that aggressive RT pressure still flattens the chronometric slope. Iterations with modest `rt_soft` weights (0.05–0.1) now log both WFPT and soft RT losses so we can dial the trade-off deliberately.

Dashboards:

- [`runs/hybrid_wfpt_curriculum_timecost_attempt1/dashboard.html`](runs/hybrid_wfpt_curriculum_timecost_attempt1/dashboard.html) — baseline time-cost guardrail (flat-slope regression)
- [`runs/hybrid_wfpt_curriculum_timecost_soft_rt/dashboard.html`](runs/hybrid_wfpt_curriculum_timecost_soft_rt/dashboard.html) — latest soft RT configuration

**Current scope (v0.1):**

- IBL-style mouse visual 2AFC task
- Macaque random-dot motion discrimination task
- Baseline agents: Sticky-Q, Bayesian observer, PPO
- Hybrid DDM+LSTM agent with stochastic evidence accumulation
- **R-DDM**: Recurrent Drift-Diffusion Model with GRU-based trial history
- Metrics, reports, and `.ndjson` logs that align agents with rodent/primate data
- **Multi-run leaderboard** (`scripts/compare_runs.py`) for cross-experiment comparison

### February 2026 — Scientific Fixes & Infrastructure

Several correctness and infrastructure improvements:

- **WFPT normalization fix**: The small-time series used incorrect image charge positions (`z + 2ka` instead of `a(z + 2k)`), causing densities to be off by a factor of `a`. Both series now agree to 6 decimal places; `drift=3, bound=2` integrates to exactly 1.000.
- **Per-trial history loss**: Root cause of the [Decoupling Problem](FINDINGS.md#the-decoupling-problem) identified — batch-mean history loss gives weak gradients, and Hybrid's history estimation was fully non-differentiable. Added `per_trial_history_loss()` with per-trial MSE supervision for both R-DDM and Hybrid trainers.
- **Ceiling-corrected chronometric slope**: When ≥2 difficulty levels are pinned at max RT, `corrected_slope` excludes ceiling levels and refits for more honest assessment.
- **Multi-run leaderboard**: `scripts/compare_runs.py` scans all runs, computes composite scores against animal reference data, outputs color-coded HTML.
- **57 tests** now pass (up from 40), including 20 WFPT tests and 12 per-trial history tests.

See [`FINDINGS.md`](FINDINGS.md) and [`PROJECT_TODO.md`](PROJECT_TODO.md) for full details.

Read the full benchmark recap in [`FINDINGS.md`](FINDINGS.md). Dashboards are stored under `runs/` for interactive inspection.

---

## Project Overview

- Provide reproducible Gymnasium environments that mirror lab protocols and timing.
- Train seeded baseline agents and log one JSON object per trial using the frozen schema.
- Run evaluation scripts that score fingerprints against shared reference datasets and render HTML reports.
- Design code paths so future PRL and DMS tasks drop in without breaking interfaces.

---

## Reference Data

AnimalTaskSim benchmarks against two canonical datasets from decision neuroscience:

### IBL Mouse 2AFC

**Source:** International Brain Laboratory (IBL) standardized protocol  
**Task:** Two-alternative forced choice with varying visual contrasts  
**Species:** Laboratory mice across multiple institutions  
**Citation:** [International Brain Laboratory (2021). *Neuron*](https://doi.org/10.1016/j.neuron.2021.04.001)

The IBL dataset provides reproducible measurements of mouse decision-making behavior with controlled contrast levels, block structure, and lapse regimes. Our environment replicates the timing, contrast levels, and block prior structure from this multi-lab effort.  
`data/ibl/reference.ndjson` now bundles 10 public IBL sessions (8,406 trials) to capture cross-session variability; the legacy single-session log remains available as `data/ibl/reference_single_session.ndjson` for reproducing earlier analyses.

### Macaque Random-Dot Motion

**Source:** Shadlen lab perceptual decision-making studies  
**Task:** Random-dot motion (RDM) direction discrimination  
**Species:** Rhesus macaques  
**Citations:**

- [Britten et al. (1992). *Journal of Neuroscience*](https://doi.org/10.1523/JNEUROSCI.12-12-04740.1992)
- [Palmer, Huk & Shadlen (2005). *Journal of Vision*](https://doi.org/10.1167/5.5.1)

The macaque RDM data captures the classic relationship between motion coherence, reaction times, and accuracy. These studies established the neural basis of evidence accumulation in area MT and inspired the drift-diffusion model framework we employ in our hybrid agent.

---

## Quickstart

### Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### Interactive Workflow (Recommended) ⭐

Run experiments with an interactive wizard that handles everything:

```bash
python scripts/run_experiment.py
```

This will guide you through:

1. **Selecting task** (IBL Mouse 2AFC or Macaque RDM)
2. **Choosing agent** (PPO, Sticky-Q, Bayesian Observer, or Hybrid DDM+LSTM)
3. **Configuring parameters** (episodes, trials, seed)
4. **Training** (with progress updates)
5. **Evaluation** (computing behavioral metrics)
6. **Dashboard generation** (interactive HTML visualization)
7. **Registry update** (automatic tracking)

**Example session:**

```text
Step 1: Select Task Environment
  [1] Mouse 2AFC (IBL) - Visual contrast discrimination
  [2] Macaque RDM - Random dot motion
Select task: 2

Step 2: Select Agent
  [1] PPO - RL baseline (~2-5 min)
  [2] Hybrid DDM+LSTM - State-of-the-art with realistic RTs (~5-15 min)
Select agent: 2

Step 3: Configure Training
Use recommended defaults? (Y/n): y

✓ Training completed!
✓ Evaluation completed!
✓ Dashboard generated!
✓ Registry updated!

Results Summary:
  Psychometric Slope: 31.75
  Chronometric Slope: -1813 ms/unit
  Win-Stay Rate: 0.490 (near chance — see FINDINGS.md)
```

### Manual Workflow (Advanced Users)

For more control, run each step separately:

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

# 5. Query your results
python scripts/query_registry.py show --run-id my_experiment
```

### Experiment Registry

Query and analyze all your experiments:

```bash
# List all experiments
python scripts/query_registry.py list

# Filter by task or agent
python scripts/query_registry.py list --task rdm_macaque --agent hybrid_ddm_lstm

# View detailed metrics for a specific run
python scripts/query_registry.py show --run-id my_experiment

# Export to CSV for analysis
python scripts/query_registry.py export --output experiments.csv
```

The registry tracks:

- Task, agent, seed, training parameters
- Behavioral metrics (psychometric, chronometric, history effects)
- File paths (config, logs, metrics, dashboard)
- Status and timestamps

Each experiment is automatically added to `runs/registry.json` for easy querying and reproducibility.

---

## Repository Layout

```text
animal-task-sim/
├─ envs/                # Gymnasium tasks + timing utilities
├─ agents/              # Sticky-Q, Bayesian observer, PPO, hybrid DDM agents
├─ animaltasksim/       # Core utilities (config, logging, seeding, registry)
├─ eval/                # Metrics, schema validator, HTML report tooling
├─ scripts/             # Train / evaluate / report CLIs (frozen interfaces)
├─ data/                # Reference animal logs and helpers
├─ tests/               # Env/agent/metric + schema unit tests
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

**Reference Data:** International Brain Laboratory (IBL) mice performing a two-alternative forced choice task with varying visual contrasts. Data from the IBL's standardized, multi-lab protocol for reproducible measurement of decision-making behavior ([International Brain Laboratory, 2021](https://doi.org/10.1016/j.neuron.2021.04.001)).

### Macaque RDM

- Motion coherence observations with optional go-cue phases.
- Actions: `left`, `right`, `hold`, with optional per-step costs.
- Supports collapsing bounds and chronometric metrics for RT alignment.

**Reference Data:** Macaque monkeys performing random-dot motion (RDM) discrimination tasks from the classic Shadlen lab studies. Animals judged motion direction of coherently moving dots with varying difficulty levels. Data derived from studies on neural correlates of perceptual decision-making in area MT and beyond ([Britten et al., 1992](https://doi.org/10.1523/JNEUROSCI.12-12-04740.1992); [Palmer, Huk & Shadlen, 2005](https://doi.org/10.1167/5.5.1)).

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

## Complete Workflow Guide

### Understanding the System

AnimalTaskSim provides a complete pipeline from training to analysis:

**Training** → **Evaluation** → **Visualization** → **Registry** → **Analysis**

Each step is automated and can be run:

- **Interactively** via `run_experiment.py` (recommended for beginners)
- **Manually** via individual scripts (for advanced users and automation)

### What Each Step Does

1. **Training (`train_agent.py`, `train_hybrid_curriculum.py`, or `train_r_ddm.py`)**
   - Trains agent on selected task environment
   - Generates trial-by-trial logs in `.ndjson` format
   - Saves model weights and training configuration
   - Output: `runs/{experiment_name}/`

2. **Evaluation (`evaluate_agent.py`)**
   - Computes behavioral metrics from trial logs
   - Fits psychometric curves (accuracy vs. evidence)
   - Computes chronometric slopes (RT vs. evidence)
   - Analyzes history effects (win-stay, lose-shift)
   - Output: `metrics.json`

3. **Dashboard (`make_dashboard.py`)**
   - Creates interactive HTML visualization
   - Compares agent vs. reference animal data
   - Plots psychometric, chronometric, and history curves
   - Output: `dashboard.html`

4. **Registry (`scan_runs.py`)**
   - Automatically detects new experiments
   - Extracts metadata and metrics
   - Updates central database (`runs/registry.json`)
   - Enables querying and comparison

5. **Query (`query_registry.py`)**
   - Filter experiments by task, agent, status
   - View detailed metrics for specific runs
   - Export to CSV for external analysis
   - Track experiment history

### File Structure After Running

```text
runs/
└── 20251017_rdm_hybrid_test/
    ├── config.json              # Training configuration
    ├── trials.ndjson            # Trial-by-trial logs (schema-validated)
    ├── metrics.json             # Behavioral metrics
    ├── evaluation.json          # Evaluation summary
    ├── dashboard.html           # Interactive visualization
    ├── model.pt                 # Trained model weights
    └── curriculum_phases.json   # (For curriculum agents)
```

### Reproducibility

Every experiment is fully reproducible via:

- **Fixed seed**: Deterministic RNG initialization
- **Saved config**: All hyperparameters stored in `config.json`
- **Schema validation**: Logs conform to frozen `.ndjson` schema
- **Registry tracking**: Complete metadata and metrics

To reproduce an experiment:

```bash
# 1. View the configuration
python scripts/query_registry.py show --run-id EXPERIMENT_NAME

# 2. Extract seed and parameters from config.json

# 3. Rerun with same parameters
python scripts/train_agent.py --seed SEED --env TASK --agent AGENT ...
```

### Best Practices

1. **Use meaningful experiment names**: Include date, task, agent, and variant
   - Good: `20251017_rdm_hybrid_curriculum_v2`
   - Bad: `test1`, `my_run`

2. **Always evaluate after training**: Metrics are needed for registry

   ```bash
   python scripts/evaluate_agent.py --run runs/YOUR_EXPERIMENT
   ```

3. **Update registry regularly**: Keep database in sync

   ```bash
   python scripts/scan_runs.py --overwrite
   ```

4. **Use dashboards for quick inspection**: Visual comparison with reference data

   ```bash
   # Dashboard path shown in registry
   python scripts/query_registry.py show --run-id YOUR_EXPERIMENT
   ```

5. **Export for deep analysis**: CSV enables statistical comparisons

   ```bash
   python scripts/query_registry.py export --output all_experiments.csv
   ```

---

## Roadmap Preview (v0.2)

### New Tasks

- **Probabilistic Reversal Learning (PRL):** bias-block reversals with perseveration metrics delivered through the same logging schema.
- **Delayed Match-to-Sample (DMS):** delay-dependent accuracy and RT metrics sharing evaluation infrastructure.

### Agent Improvements

- **Hybrid DDM+LSTM for IBL 2AFC:** Adapt the hybrid agent training pipeline to work with mouse data, including:
  - IBL reference data format compatibility
  - Hyperparameter calibration for mouse RT scales and behavioral fingerprints
  - Curriculum learning adapted for contrast levels vs. motion coherence
  - Validation against mouse history effects and block structure

### Infrastructure

- **Seed robustness tests:** Verify metric stability across 5+ seeds
- **Make-compare CLI:** One-command pipeline: train → evaluate → compare → report
- **Automated registry scans:** Ensure all runs are indexed post-sweep

---

## Acknowledgements

### Development

This project was developed independently outside of academia by with substantial contributions from AI coding assistants:

- **Claude Code (Sonnet 4.5)** - Architecture design, implementation, and documentation
- **OpenAI Codex (GPT-5-Codex)** - Code refinement and debugging support  
- **Google Gemini Pro 2.5** - Additional development assistance

This represents a collaboration between human domain expertise and AI implementation capability, demonstrating what's possible at the intersection of computational neuroscience and modern AI tools.

### Scientific Foundation

**Drift-Diffusion Modeling:**

- Roger Ratcliff & Gail McKoon for establishing the DDM framework (Ratcliff & McKoon, 2008; Ratcliff & Smith, 2016)
- Thomas Wiecki et al. for HDDM implementation and WFPT methods (Wiecki et al., 2013)
- Navarro & Fuss for numerical methods in DDM fitting (2009)

**Animal Behavioral Data:**

- **International Brain Laboratory (IBL)** - For the standardized mouse 2AFC protocol and open behavioral datasets (IBL et al., 2021, *Neuron*)
- **Shadlen Lab** - For pioneering work on macaque random-dot motion tasks and neural correlates of decision-making (Britten et al., 1992; Palmer, Huk & Shadlen, 2005)

**Behavioral Analysis:**

- Anne Urai et al. for work on choice history biases (Urai et al., 2019, *Nature Communications*)
- The broader decision neuroscience community for establishing psychometric and chronometric analysis methods

### Open Source Community

Open-source tools that made this project possible:

- **Gymnasium** - Reinforcement learning environment interface
- **PyTorch** - Deep learning framework  
- **Stable-Baselines3** - RL algorithm implementations
- **Pydantic** - Data validation and schema enforcement
- **Scientific Python ecosystem** (NumPy, SciPy, pandas, matplotlib)

### Note on Academic Affiliation

This project is **independent research** conducted outside of academic institutions. While it adheres to scientific rigor and reproducibility standards, it represents an exploration of what individuals can achieve with modern AI tools and open scientific data.

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
