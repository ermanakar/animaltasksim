# AnimalTaskSim — Project TODO

**Created**: February 2026  
**Last updated**: February 14, 2026  
**Branch**: `rddm2`

---

## Priority 1: Quick Wins

### 1.1 Fix pre-existing test failures

- [x] `tests/test_cli.py` — 2 failures: `evaluate_agent.py` NaN leak in JSON serialization (fixed: sanitize full metrics dict)
- [x] `tests/test_envs.py` — 2 failures: `rdm_macaque.py` wrote `int(correct)` instead of `bool(correct)` (fixed)
- **Why**: Broken tests erode CI trust and mask regressions.

### 1.2 README audit

- [x] Verify all accuracy/slope claims match current FINDINGS.md
- [x] Remove or qualify any "parity with animal data" language
- [x] Update test badge from 20 → 57
- [x] Add February 2026 updates section
- [x] Fix misleading "history effects consistent with reference" claim
- [x] Remove out-of-scope roadmap items (web dashboard, GPU, neural data)
- [x] Verify `pip install -e ".[dev]"` works from clean clone
- **Why**: FINDINGS.md flagged documentation drift as a risk.

---

## Priority 2: Infrastructure

### 2.1 Automated animal-vs-agent comparison dashboard

- [x] Pre-compute and cache reference metrics (`data/ibl/reference_metrics.json`, `data/macaque/reference_metrics.json`)
- [x] Build multi-run leaderboard script (`scripts/compare_runs.py`) that scans all runs, ranks by composite score
- [x] Leaderboard HTML with color-coded metric deviations, quality flags, ceiling warnings
- [ ] *(Existing)* 1:1 comparison dashboard already available via `scripts/make_dashboard.py`
- **Why**: Comparison is currently manual; every experiment should be instantly interpretable.

### 2.2 Add `make compare` CLI

- [x] `scripts/make_compare.py`: One-command pipeline: train → evaluate → compare to reference → leaderboard
- [x] Supports `--agent`, `--env`, `--episodes`, `--seed` flags
- [x] Auto-generates run directory, prints metric summary, points to leaderboard
- **Why**: Reduce friction for benchmarking new architectures.

### 2.3 Seed robustness tests

- [x] Run Sticky-Q across 5 seeds (42, 123, 7, 2024, 9999) in `tests/test_seed_robustness.py`
- [x] Verify psychometric slope CV < 1.0 across seeds
- [x] Verify bias spread < 60 across seeds
- [x] Verify win-stay/lose-shift spread < 0.5 across seeds
- [x] All seeds produce valid metrics (no crashes)
- **Why**: All current results are single-seed; stability is unknown.

---

## Priority 3: Scientific Fixes

### 3.1 Fix WFPT normalization

- [x] Diagnosed: small-time series used `z + 2ka` instead of `a(z + 2k)` for image charge positions
- [x] Fixed: both series now agree to 6 decimal places at all tested time points
- [x] Verified: drift=3, bound=2 integrates to 1.000 (was 2.4)
- [x] All 20 WFPT tests pass, including updated normalization test
- **Why**: Incorrect densities will destabilize R-DDM training as it explores wider parameter regimes.

### 3.2 RT ceiling mitigation

- [x] Option C: Ceiling-corrected chronometric slope — excludes levels pinned at max RT and refits
- [x] `corrected_slope` field added to `ChronometricMetrics` dataclass
- [x] Quality flags use `corrected_slope` when ceiling_fraction ≥ 0.5 for more honest assessment
- [x] Test: `test_ceiling_corrected_slope` validates corrected slope is computed and negative
- [ ] Option A: Document recommended wider response windows for specific agents
- [ ] Option B: Soft RT penalty (already available in Hybrid via `soft_rt_penalty`)
- **Why**: `ceiling_fraction ≥ 0.5` in flagship runs means reported slopes are artifacts.

### 3.3 Solve the Decoupling Problem

- [x] **Root cause analysed**: R-DDM's `_history_regulariser` uses batch-mean MSE → weak 1/N gradient per trial; Hybrid's `_estimate_history` uses hard argmax on detached probs → **zero gradient** for history
- [x] **Approach B — Per-trial history loss**: Added `per_trial_history_loss()` to `agents/losses.py` — operates per-trial with MSE, gives gradient proportional to each trial's deviation (strictly ≥ batch-mean by Jensen's inequality)
- [x] **R-DDM integration**: Added `per_trial_history_weight` config and wired into `_compute_losses` with same ramp schedule as existing history loss
- [x] **Hybrid integration**: Added differentiable `prob_tensor_buffer` (was `list[float]` detached), integrated per-trial loss with `no_action_value=0.0` for Hybrid encoding convention
- [x] **Tests**: 12 tests in `tests/test_per_trial_history.py` covering zero-loss, gradient flow, both conventions, edge cases, Jensen's inequality verification
- [ ] **Retrain & validate**: Retrain R-DDM and Hybrid with `per_trial_history_weight > 0`; check if win-stay/lose-shift rise from ~0.5 without regressing chronometric slope
- **Why**: This is the project's central scientific gap — no agent captures both intra-trial and inter-trial dynamics.

---

## Priority 4: Roadmap (v0.2)

### 4.1 Design PRL/DMS schema extensions

- [x] Draft additional `TrialRecord` fields: `reversal`, `block_index`, `contingency` (PRL); `sample_stimulus`, `delay_ms`, `match` (DMS)
- [x] 19 tests in `tests/test_schema_v02.py`: backward compatibility, PRL fields, DMS fields, type rejection, mixed tasks
- [x] All fields are `Optional[...] = None` — existing logs validate unchanged, `extra="forbid"` still enforced
- **Why**: v0.2 adds Probabilistic Reversal Learning and Delayed Match-to-Sample; contracts should be ready.

---

## Completed

- [x] WFPT unit tests (20 tests, `tests/test_wfpt.py`) — Feb 2026
- [x] RT ceiling detection in `eval/metrics.py` (`ceiling_fraction`, `rt_range_ms`, `rt_ceiling_warning`) — Feb 2026
- [x] R-DDM formal evaluation (`r_ddm_choice_only_v4`) — Feb 2026
- [x] AGENTS.md rewrite with concrete conventions — Feb 2026
- [x] FINDINGS.md updated with ceiling note, R-DDM eval, WFPT audit — Feb 2026
- [x] Fix 4 pre-existing test failures (`test_cli.py`, `test_envs.py`) — Feb 2026
- [x] Pre-computed reference metrics for IBL and macaque — Feb 2026
- [x] Multi-run leaderboard script (`scripts/compare_runs.py`) — Feb 2026
- [x] WFPT normalization fix (image charge positions + denominator) — Feb 2026
- [x] Per-trial history loss (`agents/losses.py`, 12 tests) — Feb 2026
- [x] R-DDM + Hybrid trainer integration for per-trial history loss — Feb 2026
- [x] Seed robustness tests (5 seeds, `tests/test_seed_robustness.py`) — Feb 2026
- [x] PRL/DMS schema extensions (6 fields, 19 tests, `tests/test_schema_v02.py`) — Feb 2026
