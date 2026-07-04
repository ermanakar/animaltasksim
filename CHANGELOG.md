# Changelog

All notable changes to AnimalTaskSim are documented here. This file tracks
**releases and contract-affecting changes** — CLI arguments, schema keys, file
paths, environment behavior — plus headline features. For the scientific record
(experiments, metrics, negative results, target provenance) see
[FINDINGS.md](FINDINGS.md); this changelog deliberately does not duplicate it.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added

- **Adaptive-control agent**: a new agent archetype that adds persistence, exploration, and uncertainty-gated arbitration control on top of the hybrid evidence core (`agents/adaptive_control_*.py`, `scripts/train_adaptive_control.py`, `--control-profile {no_control, persistence_only, exploration_only, full_control}`). `persistence_only` is the validated default; the others are comparison/lesion conditions.
- **Adaptive-control probe metrics** (`eval/metrics.py`): retry gap, stale-switch lift, unrewarded/volatile-switch lift, and block-switch adaptation lift, plus the matched validation-suite and interaction-sweep scripts (`scripts/adaptive_control_validation_suite.py`, `scripts/adaptive_control_interaction_sweep.py`, `scripts/adaptive_control_persistence_sweep.py`).
- **Probabilistic Reversal Learning transfer path**: hidden-contingency PRL environment (`envs/prl_reversal.py`), PRL reversal metrics, HTML report visualization, registry metadata, and a matched transfer suite.
- **Delayed Match-to-Sample scaffold**: schema-valid DMS environment (`envs/dms_match.py`) plus tests and a memory-fingerprint design note. Adaptive-control DMS rollout is intentionally unwired.
- **Schema v0.2 optional fields** (`eval/schema_validator.py`, additive; `extra="forbid"` preserved): PRL (`reversal`, `block_index`, `contingency`) and DMS (`sample_stimulus`, `delay_ms`, `match`) trial fields. Registry gains PRL metadata and `prl`/`dms` task values.
- **Adaptive-control lesion / recurrence flags**: `uncertain_retry_enabled` (default on) and the flag-gated change-evidence recurrence (`change_evidence_enabled` / `change_evidence_decay`, default off; verified flag-off bit-for-bit no-op). λ=0.9 is the validated opt-in cross-task profile.
- **PRL arbitration diagnostic**: a checkpoint-reroll CLI writing a separate `control_diagnostics.ndjson` sidecar, leaving the frozen trial schema unchanged.
- **Reproducible IBL reference fetcher** (`scripts/fetch_ibl_reference.py`): pulls multi-session `biasedChoiceWorld` data from the IBL public server (OpenAlyx) into the project schema, with convention-agnostic action derivation, choice-sign auto-calibration, a trained-performance QC gate, and an EID manifest. Requires `ONE-api` (kept out of `pyproject.toml`). Add-and-compare only; `reference.ndjson` is unchanged.
- Regression coverage across PRL / DMS / registry / diagnostics / change-evidence / fetcher. Total: 185 tests.

### Changed

- **Adaptive-control default** is `persistence_only` (validated); full control is a comparison condition.
- **Registry task inference**: the config `task` is authoritative over run-directory-name heuristics.
- **IBL fetcher RT source** defaults to `response_times - stimOn` (matches the baseline `reference.ndjson`); `firstMovement` is available but does not match the project's calibrated chronometric targets.
- **Documentation**: FINDINGS.md restructured into a navigable chronological record (full raw notebook archived under `docs/archive/`).

### Fixed

- **Adaptive retry-gap provenance**: `compute_adaptive_control_probe_metrics()` bins retry by the *prior failed* trial's stimulus strength, not the newly sampled current trial.

_Scientific results and validation numbers for all of the above live in [FINDINGS.md](FINDINGS.md)._

## [0.2.1] - 2026-03-01

### Fixed

- **`prev_reward` rollout bug (critical)**: the outcome-phase check used `phase_step == 0`, but the environment returns reward *before* advancing the phase step and returns info *after* — corrected to `== 1`. `prev_reward` had always been 0.0, routing every trial through the lose-history pathway and pinning win-stay near 0.556. This single bug explained why prior history interventions had failed.
- **Mixed-provenance behavioral targets**: all IBL targets now derived from per-session analysis of `data/ibl/reference.ndjson` (10 sessions, 8,406 trials); previously assembled from three independent sources.
- **IBL contrast set**: removed the 0.5 contrast from `envs/ibl_2afc.py` (not part of the biased-blocks protocol).
- **Reporting**: agent results now reported as 5-seed mean ± std instead of a single best seed.

### Added

- **Co-evolution training** for the Hybrid agent: training from scratch with history injection active, so the evidence circuits learn alongside the history effects (post-hoc injection degraded psychometric slope). `drift_magnitude_target` recalibrated 6.0 → 9.0 to compensate; produces all six IBL behavioral fingerprints simultaneously (5-seed validation). Adds `scripts/run_5seed_validation.py` and history diagnostic/sweep scripts.
- `scripts/compute_reference_targets.py`: per-session psychometric / chronometric / history metrics → `data/ibl/reference_targets.json`, the single source of truth for behavioral targets.

### Changed

- Regenerated `data/ibl/metrics.json` from the multi-session reference.
- Documented chronometric-target provenance in `eval/metrics.py`.

## [0.2.0] - 2026-02-19

### Added

- **Separate history-network architecture**: an MLP bypass path (2→8→1, zero-init) computing `stay_tendency` from (prev_action, prev_reward), independent of the LSTM evidence path.
- **Drift-rate bias mechanism**: history modulates evidence accumulation throughout the trial, producing history effects at all difficulty levels (unlike starting-point bias, which only affects ambiguous trials).
- **IBL 2AFC support for the Hybrid agent**: `train_hybrid_curriculum.py --task ibl_2afc` with auto-inferred reference data and IBL phase timing.
- **Per-trial history loss** (`agents/losses.py`) and **multi-seed validation** (`scripts/seed_sweep.py`).
- 14 new tests (history architecture, seed robustness, schema v0.2). Total: 93 tests.

### Fixed

- **WFPT image charge positions**: small-time series corrected to `z + 2ka`; both series now agree to 6 decimal places.
- **Bias measurement artifact**: the "84% leftward bias" was holds counted in the denominator; added the `p_right_committed` metric that excludes holds.
- **DDM timing alignment**: `max_commit_steps` matched to the environment response phase, preventing artificial hold rates.

### Changed

- Curriculum expanded to 7 phases; loss-weight defaults rebalanced (WFPT-dominant early).

### Removed

- Superseded scripts (`train_hybrid.py`, `pretrain_hybrid.py`, `finetune_hybrid.py`, `calibration.py`, `eval_wfpt.py`, `make_compare.py`, `experiment_decoupling.py`, debug scripts), the duplicate `reference_multi_session.ndjson`, and root-level one-off scripts.

## [0.1.2] - 2025-10-12

### Added

- Curriculum-learning framework for the hybrid agent (WFPT-based default; phases can override commit windows).

### Changed

- **Reward structure**: an incorrect choice is now −0.1 (was 0.0).
- **Observation space**: the environment can now include the previous trial's action, reward, and correctness.
- **Zero-contrast trials**: the hard-coded logic was removed; the agent is rewarded for choosing the higher-prior side.

## [0.1.1] - 2025-10-11

### Added

- **WFPT (Wiener First Passage Time) likelihood loss** (`agents/wfpt_loss.py`): p(choice, RT | DDM params) with small- and large-time series approximations.
- **Drift-magnitude regularization** and **mini-batch training** for the hybrid agent; `drift_magnitude` exposed in the CLI.

### Fixed

- Dashboard chronometric/accuracy plots auto-detect contrast vs coherence (adds IBL support).
- Chronometric metrics are now computed for the IBL 2AFC task.
- **Collapsing-bound override bug**: the environment's `collapsing_bound=True` overrode the agent's learned commit step; delegating timing to the agent's DDM restored RT dynamics.

## [0.1.0] - 2025-10-10

### Initial release

- **Environments**: IBL 2AFC (`envs/ibl_2afc.py`), macaque RDM (`envs/rdm_macaque.py`), shared timing utilities.
- **Agents**: Sticky-Q, Bayesian observer, PPO (action masking), DDM baseline, Hybrid DDM+LSTM.
- **Evaluation**: psychometric/chronometric/history metrics, HTML reports, comparison dashboards, NDJSON schema validator.
- **Scripts**: train (baseline + hybrid curriculum), evaluate, report, dashboard, interactive experiment wizard.
- **Data**: IBL mouse reference (10 sessions / 8,406 trials) + legacy single-session; macaque RDM reference (2,611 trials).
- **Infrastructure**: Python 3.11 typing, deterministic seeding, schema-validated NDJSON logging, CPU-only (<20 min, <4 GB RAM), frozen CLI interfaces.
- Tests: environments, agents, metrics, schema, CLI.

---

## Format

**Added** new features · **Changed** existing behavior · **Deprecated** soon-to-be-removed · **Removed** now-removed · **Fixed** bug fixes · **Security** vulnerabilities
