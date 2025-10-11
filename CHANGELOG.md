# Changelog

All notable changes to AnimalTaskSim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.1] - 2025-10-11

### Dashboard Plotting Fixed for IBL Task

#### Fixed

- **Dashboard chronometric and accuracy plots now support both contrast and coherence** (`eval/dashboard.py`)
  - Previously `_plot_chronometric_comparison()` and `_plot_accuracy_by_coherence()` only checked for `stimulus_coherence`
  - IBL mouse task uses `stimulus_contrast`, causing plots to show "No coherence data" error
  - Both functions now auto-detect stimulus column (contrast or coherence)
  - Use absolute values for difficulty binning (handles signed contrast ±1.0 and unsigned coherence 0-0.512)
  - Label axes appropriately: "|Contrast|" for IBL, "|Coherence|" for macaque
  - Result: All 4 plots (psychometric, chronometric, history, accuracy) now render correctly for both tasks

### Chronometric Analysis Enabled for IBL Task

#### Fixed

- **Chronometric metrics now computed for IBL mouse 2AFC task** (`eval/metrics.py:218`)
  - Previously only psychometric and history metrics were computed despite RT data being present
  - Added `compute_chronometric(df, stimulus_key="contrast")` call for IBL task
  - Function already supported both contrast (signed, ±1.0) and coherence (unsigned, 0-0.512) via `abs()` conversion
  - IBL reference results: slope -36.43 ms/unit, intercept 300.0 ms, RT range 260-319 ms

### Hybrid DDM+LSTM Agent - WFPT Training Implementation

#### Added

- **WFPT (Wiener First Passage Time) likelihood loss** (`agents/wfpt_loss.py`)
  - Likelihood-based DDM training objective: p(choice, RT | drift, bound, bias, noise, non_decision)
  - Small-time and large-time series approximations for numerical stability
  - Gradient propagation through all DDM parameters
  - 271 lines including test harness

- **Drift magnitude regularization** (`agents/losses.py`, `agents/hybrid_ddm_lstm.py`)
  - Regularization term: `(drift_gain - 12)²` with configurable weight
  - Anchors parameter scale to target regime (drift_gain 10-20)
  - Prevents convergence to weak-drift local minima observed in Attempt 10
  - Integrated into training metrics and CLI arguments

- **Mini-batch training** (`agents/hybrid_ddm_lstm.py:224-259`)
  - Automatic splitting of large sessions into batches of 100 trials
  - Increases gradient updates: 26 batches per epoch (520 total) vs 1 batch (15 total)
  - Required for sufficient optimization steps on single-session datasets

#### Fixed

- **Collapsing bound override bug** (`agents/hybrid_ddm_lstm.py:661`)
  - Environment's `collapsing_bound=True` triggered auto-commit at ~6 steps based on internal evidence
  - Agent's learned commit_step_target (48-120 steps) was ignored during rollout
  - All RTs collapsed to 60-80ms minimum regardless of learned DDM parameters
  - Fix: Set `collapsing_bound=False` to delegate timing control to agent's DDM
  - Measured impact: RTs 76ms → 1103ms, slope -0.27 → -169 ms/unit in diagnostic test

- **Type annotations in debug scripts**
  - `debug_ddm_rollout.py`: Corrected device parameter type (str → torch.device)
  - `debug_rollout.py`: Added type: ignore for info dict access

#### Changed

- Loss tracking: Added `epoch_drift_magnitude` to training metrics
- Configuration: drift_magnitude parameter exposed in CLI (scripts/train_hybrid.py)
- Documentation: Updated with diagnostic process and quantitative results

### Results (Attempt 11: `runs/rdm_wfpt_regularized/`)

**Chronometric (RT Dynamics):**

- Slope: -981 ms/unit (macaque reference: -645 ms/unit, ratio 1.52)
- RT range: 710ms (high coherence) → 1200ms (low coherence)
- Inverted-U chronometric curve shape

**Psychometric (Choice Behavior):**

- Slope: 10.93 (macaque: 17.56, ratio 0.62)
- Bias: -0.001 (macaque: +0.0003, both ~0)
- Lapses: ~10^-13 (macaque: ~10^-16, both negligible)

**History Effects:**

- Win-stay: 0.50 (macaque: 0.46)
- Lose-shift: 0.50 (macaque: 0.52)
- Sticky-choice: 0.50 (macaque: 0.46)

**DDM Parameters:**

- drift_gain: 12-18 (target regime)
- SNR: 0.029 → 0.396 (coherence-dependent scaling)
- bounds: 1.9-2.7

**Training Configuration:**

```python
LossWeights(choice=1.0, rt=0.0, wfpt=1.0, history=0.1, drift_magnitude=0.5)
```

20 epochs, 26 batches/epoch, 520 total gradient updates. Seed 43, training time ~20min CPU.

### Documentation

- Updated `README.md` with quantitative results and technical approach
- Updated `FINDINGS.md` with Attempt 11 analysis and historical context
- Updated `TRAINING_PROGRESS.md` with attempt summary and limitations
- Created `BUGFIX_SUMMARY.md` documenting collapsing bound diagnostic process
- Created `CHANGELOG.md` (this file) for version tracking

### Limitations

- Low coherences (0.0-0.128) reach 1200ms timeout (macaque: 660-760ms)
- RT intercept elevated: 1259ms vs 759ms (macaque), +500ms offset
- Psychometric slope shallower: 10.93 vs 17.56 (62% of reference)
- Calibration targets for future work: non-decision time, bound parameters, timeout window

---

## [0.1.0] - 2025-10-10

### Initial Release

**Added:**

- **Environments**
  - `envs/ibl_2afc.py`: IBL-style mouse visual 2AFC task
  - `envs/rdm_macaque.py`: Macaque random-dot motion discrimination
  - `envs/utils_timing.py`: Shared timing utilities

- **Agents**
  - `agents/sticky_q.py`: Sticky Q-learning baseline
  - `agents/bayes_observer.py`: Bayesian observer with sensory noise
  - `agents/ppo_baseline.py`: PPO with action masking
  - `agents/ddm_baseline.py`: Drift Diffusion Model baseline
  - `agents/hybrid_ddm_lstm.py`: Hybrid DDM+LSTM agent (initial implementation)

- **Evaluation Stack**
  - `eval/metrics.py`: Psychometric, chronometric, history metrics
  - `eval/report.py`: HTML report generation
  - `eval/dashboard.py`: Interactive comparison dashboards
  - `eval/schema_validator.py`: NDJSON log validation

- **Scripts**
  - `scripts/train_agent.py`: Train baseline agents
  - `scripts/train_hybrid.py`: Train hybrid DDM+LSTM agent
  - `scripts/evaluate_agent.py`: Compute metrics
  - `scripts/make_report.py`: Generate HTML reports
  - `scripts/make_dashboard.py`: Generate comparison dashboards
  - `scripts/calibration.py`: Hyperparameter optimization

- **Data**
  - `data/ibl/reference.ndjson`: Mouse 2AFC reference data (885 trials)
  - `data/macaque/reference.ndjson`: Macaque RDM reference data (2611 trials)

**Testing:**

- `tests/test_envs.py`: Environment unit tests
- `tests/test_agents.py`: Agent unit tests
- `tests/test_metrics.py`: Metrics unit tests
- `tests/test_schema.py`: Schema validation tests
- `tests/test_cli.py`: CLI tests

**Documentation:**

- `README.md`: Project overview, quickstart, task descriptions
- `PRD.md`: Product requirements document
- `FINDINGS.md`: Benchmark results and insights
- `AGENTS.md`: Agent operating guide
- `kickoff_prompt.md`: Project kickoff specification

**Infrastructure:**

- Python 3.11 type hints and docstrings
- Deterministic seeding (Python, NumPy, PyTorch)
- Schema-validated NDJSON logging with line-by-line flushing
- CPU-only training (<20 min, <4 GB RAM for demos)
- Frozen CLI interfaces for reproducibility

---

## Format

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** in case of vulnerabilities
