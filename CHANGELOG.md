# Changelog

All notable changes to AnimalTaskSim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] - 2026-02-19

### Added

- **Separate history network architecture**: MLP bypass path (2→8→1, ReLU, zero-init) that computes `stay_tendency` from (prev_action, prev_reward), independent of LSTM evidence processing. Mirrors biological separation between PFC/basal ganglia (history) and LIP/parietal (evidence).
- **Drift-rate bias mechanism**: History modulates evidence accumulation throughout the trial via `drift = gain × stimulus + stay_tendency × drift_scale × prev_direction`. This produces history effects at all difficulty levels, unlike starting-point bias which only affects ambiguous trials.
- **IBL 2AFC support for Hybrid agent**: `train_hybrid_curriculum.py` now supports `--task ibl_2afc` with auto-inferred reference data, contrast-based stimulus processing, and IBL-specific phase timing.
- **Per-trial history loss** (`agents/losses.py`): Supervises P(stay) on every individual trial rather than batch-level aggregation, avoiding Jensen's inequality artifacts.
- **Multi-seed validation** (`scripts/seed_sweep.py`): Runs 5 seeds sequentially, computes mean ± std across seeds, writes structured results to `sweep_results.json`.
- **14 new tests**: History architecture tests (`test_history_bias_head.py`, `test_per_trial_history.py`), seed robustness tests (`test_seed_robustness.py`), schema v0.2 tests (`test_schema_v02.py`). Total: 93 tests.

### Fixed

- **WFPT image charge positions**: Corrected small-time series to use `z + 2ka` (not `z + 2k`). Both series now agree to 6 decimal places.
- **Bias measurement artifact**: Discovered that "84% leftward bias" was holds counted in denominator. Added `p_right_committed` metric that excludes holds. All prior Hybrid conclusions reassessed.
- **DDM timing alignment**: Response duration override ensures `max_commit_steps` matches environment response phase, preventing artificial hold rates.

### Changed

- **Curriculum updated to 7 phases**: WFPT warmup → gentle choice → annealed choice → history supervision → RT calibration → RT weighting → history finetune (frozen DDM, only history_network trains).
- **Loss weight defaults**: WFPT dominant in early phases, per-trial history loss in Phase 7 with separate learning rate (3e-3 vs 1e-3 main).

### Results (IBL 2AFC, 5-seed validation)

| Metric | Agent (mean ± std) | IBL Mouse | Match |
|--------|-------------------|-----------|-------|
| Win-stay | 0.665 ± 0.015 | 0.724 | 92% |
| Lose-shift | 0.405 ± 0.016 | 0.427 | 95% |
| Chrono slope | -66.7 ± 2.0 ms/unit | negative | direction ✓ |
| Psych slope | 6.31 ± 0.38 | ~13.2 | 48% |
| Commit rate | 100% | 100% | exact |

Config: `--task ibl_2afc --history-drift-scale 15.0 --episodes 30 --max-sessions 80`

### Removed

- Superseded scripts: `train_hybrid.py`, `pretrain_hybrid.py`, `finetune_hybrid.py`, `calibration.py`, `eval_wfpt.py`, `make_compare.py`, `generate_readme_figures.py`, `experiment_decoupling.py`, debug scripts.
- Duplicate data: `data/ibl/reference_multi_session.ndjson` (identical to `reference.ndjson`).
- Root-level one-off scripts: `count_actions.py`, `analyze_rollout.py`.

---

## [0.1.2] - 2025-10-12

Added

- **Curriculum Learning Framework**: Implemented a curriculum learning framework for the hybrid DDM+LSTM agent.
- **WFPT-based Curriculum**: Created a new default curriculum that prioritizes the WFPT loss in the initial phase of training.
- **Curriculum extensions**: Curriculum phases can now override commit windows; `CurriculumConfig.wfpt_history_refine()` adds an optional history-focused third phase, while `CurriculumConfig.wfpt_time_cost()` introduces a time-cost constrained schedule for RT tuning experiments.
- **Hybrid timing guardrails**: Increased the hybrid agent’s default `max_commit_steps` to 180, bolstered the WFPT warm-up (longer phase, higher drift/non-decision supervision), and rebalanced the time-cost curriculum so WFPT remains dominant while RT penalties stay gentle.

### Changed

- **Default Curriculum**: The default curriculum for the hybrid agent is now a two-phase, WFPT-based curriculum.
- **Reward Structure**: The reward for an incorrect choice is now -0.1 (previously 0.0).
- **Observation Space**: The environment now supports including the previous trial's action, reward, and correctness in the observation.
- **Zero-Contrast Trials**: The hard-coded logic for zero-contrast trials has been removed. The agent is now rewarded for choosing the side with the higher prior probability.

### Results (`runs/hybrid_wfpt_curriculum/`)

**Chronometric (RT Dynamics):**

- Slope: -767 ms/unit (macaque reference: -645 ms/unit)
- RT range: 790ms (high coherence) → 1200ms (low coherence)

**Psychometric (Choice Behavior):**

- Slope: 7.33 (macaque: 17.56)
- Bias: +0.001 (macaque: ≈0)

## [0.1.1] - 2025-10-11

### Added

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

### Fixed

- **Dashboard Plotting for IBL Task**: Dashboard chronometric and accuracy plots now support both contrast and coherence (`eval/dashboard.py`).
  - Previously `_plot_chronometric_comparison()` and `_plot_accuracy_by_coherence()` only checked for `stimulus_coherence`, causing an error for the IBL mouse task which uses `stimulus_contrast`.
  - Both functions now auto-detect the stimulus column and use absolute values for binning, correctly rendering all plots for both tasks.

- **Chronometric Analysis for IBL Task**: Chronometric metrics are now computed for the IBL mouse 2AFC task (`eval/metrics.py:218`).
  - Previously, only psychometric and history metrics were computed. The fix adds a `compute_chronometric(df, stimulus_key="contrast")` call.

- **Collapsing bound override bug in Hybrid DDM+LSTM Agent** (`agents/hybrid_ddm_lstm.py:661`):
  - The environment's `collapsing_bound=True` was overriding the agent's learned commit step, causing all RTs to collapse to a minimum.
  - Setting `collapsing_bound=False` delegates timing control to the agent's DDM, restoring correct RT dynamics.

- **Type annotations in debug scripts**: Corrected type hints in debug scripts (since removed in v0.2.0).

### Changed items

- **Hybrid DDM+LSTM Agent**:
  - Loss tracking now includes `epoch_drift_magnitude`.
  - The `drift_magnitude` parameter is exposed in the CLI.
  - Documentation was updated with diagnostic process and results.

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
  - `scripts/train_hybrid_curriculum.py`: Train hybrid DDM+LSTM agent
  - `scripts/evaluate_agent.py`: Compute metrics
  - `scripts/make_report.py`: Generate HTML reports
  - `scripts/make_dashboard.py`: Generate comparison dashboards
  - `scripts/run_experiment.py`: Interactive experiment wizard

- **Data**
- `data/ibl/reference.ndjson`: Mouse 2AFC reference data (multi-session aggregate; 10 sessions, 8,406 trials)
- `data/ibl/reference_single_session.ndjson`: Legacy single-session reference (885 trials)
  - `data/macaque/reference.ndjson`: Macaque RDM reference data (2611 trials)

**Testing:**

- `tests/test_envs.py`: Environment unit tests
- `tests/test_agents.py`: Agent unit tests
- `tests/test_metrics.py`: Metrics unit tests
- `tests/test_schema.py`: Schema validation tests
- `tests/test_cli.py`: CLI tests

**Documentation:**

- `README.md`: Project overview, quickstart, task descriptions
- `FINDINGS.md`: Benchmark results and insights
- `AGENTS.md`: Agent operating guide

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
