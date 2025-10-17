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

## Reference Data

AnimalTaskSim benchmarks against two canonical datasets from decision neuroscience:

### IBL Mouse 2AFC

**Source:** International Brain Laboratory (IBL) standardized protocol  
**Task:** Two-alternative forced choice with varying visual contrasts  
**Species:** Laboratory mice across multiple institutions  
**Citation:** [International Brain Laboratory (2021). *Neuron*](https://doi.org/10.1016/j.neuron.2021.04.001)

The IBL dataset provides reproducible measurements of mouse decision-making behavior with controlled contrast levels, block structure, and lapse regimes. Our environment replicates the timing, contrast levels, and block prior structure from this multi-lab effort.

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
  Win-Stay Rate: 0.490
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

1. **Training (`train_agent.py` or `train_hybrid_curriculum.py`)**
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

- **Probabilistic Reversal Learning (PRL):** bias-block reversals with perseveration metrics delivered through the same logging schema.
- **Delayed Match-to-Sample (DMS):** delay-dependent accuracy and RT metrics sharing evaluation infrastructure.

---

## License

Code is released under MIT; datasets retain their original licenses.
