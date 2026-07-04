# FINDINGS

**Benchmarking reinforcement-learning agents against rodent and primate decision-making fingerprints**
*October 2025 – July 2026 · from v0.1.0 to the v0.2 adaptive-control/PRL/DMS scaffold*

## Current State (July 2026)

The **Decoupling Problem is architecturally solved**: the Hybrid DDM+LSTM (differentiable Euler-Maruyama DDM, asymmetric win/lose history networks, drift-rate bias, attention-gated history, fixed rollout lapse, co-evolution training) produces all six IBL behavioral fingerprints simultaneously. 5-seed co-evolution validation (win_t=0.30, lose_t=0.15, drift_magnitude_target=9.0): psych slope 12.38 ± 0.64, chrono slope -34.2 ± 1.8 ms/unit, win-stay 0.706 ± 0.008, lose-shift 0.457 ± 0.007, lapse ~0.075. History effects, chrono, lose-shift, and lapse fall within the reference per-session range; psych slope sits below the reference mean. History effects currently use **injected fixed values**, not values learned by the networks — that remains the open frontier.

For **adaptive control**, `persistence_only` is the validated/default IBL profile (uncertainty-gated weak-failure retry, 5/5 seeds). Exploration is not independently validated on stable IBL. On **PRL** (hidden-contingency reversal), the deficit was localized to the `uncertain_retry` term firing at full strength because neutral options pin perceptual uncertainty at 1.0; the flag-gated **change-evidence recurrence** fixes the mechanism (verified flag-off bit-for-bit no-op). Safety-gated calibration rejected λ=0.7 and selected **λ=0.9 as the validated opt-in cross-task profile** (feature default off). The **DMS** memory fingerprint is defined but not yet wired for adaptive rollout. The **IBL reference expansion** (80 QC'd sessions, 57,888 trials) independently reproduces all six fingerprints under the correct `response_times` RT convention, but is add-and-compare only — the frozen 10-session `reference.ndjson` and its targets remain canonical.

## How to read this document

The story runs chronologically through six arcs. **(1) Early baselines (Sept–Oct 2025)** establish the Decoupling Problem and the three agent archetypes (reward optimizers, history heuristics, mechanistic integrators), including the long PPO calibration chronicle of negative results. **(2) The Decoupling experiments (Feb 2026)** chase per-trial history losses, discover that the "84% leftward bias" was a metric artifact, and discover the Roitman & Shadlen macaque has no history effects — the target didn't exist. **(3) The architectural solution (Feb 2026)** switches to IBL mouse data and finds that drift-rate bias + attention-gating + a differentiable DDM simulator produce all fingerprints. **(4) The prev_reward bug and co-evolution (Feb–Mar 2026)** show a single rollout bug invalidated all history analysis, and co-evolution training finally calibrates every fingerprint at once; plastic-history attempts then reframe the problem toward an adaptive control system. **(5) Adaptive control, PRL, and change-evidence (May–June 2026)** lesion-test a persistence/exploration/arbitration controller, transfer it to hidden-contingency reversal learning, and resolve a perseveration mechanism. **(6) DMS and IBL reference expansion (June–July 2026)** define the memory scorecard and independently cross-validate the reference fingerprint at scale. **Read the Methodological Notes first** — two dated reframes (target provenance; RT-ceiling saturation) change how numbers throughout should be read.

## Table of Contents

- [Methodological Notes That Reframe the Numbers](#methodological-notes-that-reframe-the-numbers)
- [Early Baselines and the Decoupling Problem (Sept–Oct 2025)](#early-baselines-and-the-decoupling-problem-septoct-2025)
- [PPO Calibration Chronicle — Negative Results (Sept–Oct 2025)](#ppo-calibration-chronicle--negative-results-septoct-2025)
- [Three Agent Archetypes and the Path Forward (Oct 2025)](#three-agent-archetypes-and-the-path-forward-oct-2025)
- [Infrastructure and Loss Audits (Feb 2026)](#infrastructure-and-loss-audits-feb-2026)
- [The Decoupling Experiment and the Bias Artifact (Feb 2026)](#the-decoupling-experiment-and-the-bias-artifact-feb-2026)
- [Solving the Decoupling Problem: Drift-Rate Bias, Attention Gating, Differentiable DDM (Feb 2026)](#solving-the-decoupling-problem-drift-rate-bias-attention-gating-differentiable-ddm-feb-2026)
- [The prev_reward Bug and Co-Evolution Training (Feb 2026)](#the-prev_reward-bug-and-co-evolution-training-feb-2026)
- [Plastic History and the Adaptive-Control Reframing (March 2026)](#plastic-history-and-the-adaptive-control-reframing-march-2026)
- [Adaptive Control Experiments (May 2026)](#adaptive-control-experiments-may-2026)
- [Reporting and Figures Pass (May 2026)](#reporting-and-figures-pass-may-2026)
- [PRL Transfer and the Perseveration Mechanism (May 30, 2026)](#prl-transfer-and-the-perseveration-mechanism-may-30-2026)
- [Change-Evidence Recurrence and Cross-Task Calibration (May 31 – June 1, 2026)](#change-evidence-recurrence-and-cross-task-calibration-may-31--june-1-2026)
- [Adaptive Retry Metric Provenance Correction (June 1, 2026)](#adaptive-retry-metric-provenance-correction-june-1-2026)
- [DMS Memory Fingerprint Defined (June 1, 2026)](#dms-memory-fingerprint-defined-june-1-2026)
- [IBL Reference Expansion (July 2026)](#ibl-reference-expansion-july-2026)

---

## Methodological Notes That Reframe the Numbers

Two dated notes are pinned first because they reframe numbers throughout the rest of the document.

### Target Provenance Correction (March 2026)

**Behavioral targets were previously derived from mixed sources.** Prior to March 2026, the documented IBL targets were assembled from three independent sources — an idealized "Frankenstein mouse" no single dataset shows:

- **Psychometric slope (13.2):** from a single IBL session (885 trials, `reference_single_session.ndjson`), NOT part of the 10-session aggregate. **Superseded.**
- **History effects (WS=0.724, LS=0.427):** from the 10-session aggregate (8,406 trials, `reference.ndjson`).
- **Chronometric slope (-36 ms/unit):** approximate IBL literature value; neither reference file produces it (single-session: -3.6, aggregate: -15.6).

Additionally, the IBL env included a **0.5 contrast** not in the biased-blocks protocol (all 10 reference sessions use only {0, 0.0625, 0.125, 0.25, 1.0}). **Corrected targets** now come from per-session analysis of `reference.ndjson` (10 sessions, 8,406 trials; one degenerate session with psych=200 excluded from psychometric stats):

| Metric | Per-session mean ± std | Per-session median | Aggregate (pooled) |
|--------|----------------------|-------------------|-------------------|
| Psych slope | 20.0 ± 5.7 (n=9) | 20.3 | 21.1 |
| Chrono slope | -51.0 ± 63.7 (n=9) | -44.3 | -15.6 |
| Win-stay | 0.72 ± 0.08 (n=10) | 0.70 | 0.725 |
| Lose-shift | 0.47 ± 0.10 (n=10) | 0.50 | 0.434 |
| Lapse low | 0.08 ± 0.07 (n=10) | 0.07 | 0.08 |
| Lapse high | 0.10 ± 0.14 (n=10) | 0.07 | 0.12 |

**Chrono caveat:** per-session chrono slopes have enormous variance (range -2 to -202 ms/unit) from ~500–1000 trials/session. The literature -36 ms/unit is retained in `eval/metrics.py` for continuity but interpreted cautiously.

**Agent-result caveat:** previous docs reported the best single-seed result (seed 42: psych=13.16). The 5-seed validation (seeds 42, 123, 456, 789, 1337) gives psych=12.38 ± 0.64, chrono=-34.2 ± 1.8, WS=0.706 ± 0.008, LS=0.457 ± 0.007. All subsequent reporting uses 5-seed mean ± std. Historical Phase 1–16 tables retain their **original target comparisons** to preserve the narrative. See `scripts/compute_reference_targets.py` and `data/ibl/reference_targets.json`.

### RT Ceiling Saturation (February 2026)

Several runs report large negative chronometric slopes (e.g., −1813 ms/unit for `20251019_rdm_hybridddml`), but the slope is driven by a **step function** between ceiling-clamped slow trials and fast high-coherence trials, not smooth evidence accumulation. Two metrics now flag this:

- **`ceiling_fraction`**: fraction of difficulty levels where median RT equals the maximum observed RT. **Values ≥ 0.5 indicate the slope is unreliable.**
- **`rt_range_ms`**: range between fastest and slowest median RTs.

| Run | Reported slope | `ceiling_fraction` | RT distribution |
| --- | --- | --- | --- |
| `20251019_rdm_hybridddml` | −1813 ms/unit | **0.50** (3/6 at 1200 ms) | Step: 1200/1200/1200/880/540/370 |
| `hybrid_wfpt_curriculum` | −767 ms/unit | ~0.50 | Similar ceiling pattern |
| Macaque reference | −645 ms/unit | 0.0 | Graded 302–525 ms, no ceiling |

**Takeaway:** when `ceiling_fraction ≥ 0.5`, the slope reflects response-window limits, not accumulation. True DDM-like chronometry requires RTs varying smoothly across all difficulty levels without hitting environment ceilings.

---

## Early Baselines and the Decoupling Problem (Sept–Oct 2025)

The October 2025 round hardened the pipeline (remove simulator shortcuts — auto-commit, implicit latency) and regenerated baselines to document the resulting behavioral gaps honestly.

### Infrastructure changes

- **Collapsing bound default off**: `envs/rdm_macaque.py` defaults to `collapsing_bound=False`, forcing agents to manage commitment timing.
- **Mouse latency hook**: `envs/ibl_2afc.py` exposes `min_response_latency_steps`; agents cannot act until latency expires, but the env no longer commits for them.
- **Config persistence**: Sticky-Q and PPO serialize latency/reward-tuning fields into each `config.json`.
- **JSON hygiene**: `evaluate_agent.py` / `eval/metrics.py` coerce NaN fits to `null`, keeping `metrics.json` schema-compliant and flagging failed regressions explicitly.

These ensure RT metrics reflect agent policy, not environment defaults.

### Sticky-Q with latency (IBL) — `runs/ibl_stickyq_latency/`

`min_response_latency_steps=20` (≈200 ms), seed 0.

| Metric | Agent | Reference (IBL) | Gap |
| --- | --- | --- | --- |
| Bias | −0.0001 | +0.074 | Matches magnitude, opposite sign |
| Psychometric slope | 33.3 | 13.2 | 2.5× steeper |
| Median RT | 210 ms | 300 ms | 90 ms faster despite added latency |
| RT slope (ms/unit) | −0.2 | −36.4 | Essentially flat |
| Win-stay | 0.67 | 0.73 | Under-expresses |
| Lose-shift | 0.48 | 0.34 | Overreacts to errors |

Explicit latency lifts RT intercepts but produces no coherence-dependent curve; Sticky-Q executes immediately when the gate opens. Choice slope too steep; sequential dependencies asymmetric (over-perseverates on losses, under-perseverates on wins).

### PPO baseline (RDM) — `runs/rdm_ppo_latest/`

`collapsing_bound=False`, 200k timesteps.

| Metric | Agent | Reference (Roitman & Shadlen) | Gap |
| --- | --- | --- | --- |
| Bias | +0.52 | ≈0 | Large pathological bias |
| Psychometric slope | 50.0 | 17.6 | 2.8× steeper |
| Median RT | 60 ms | 760 ms | 700 ms too fast |
| RT slope | 0 ms/unit | −645 ms/unit | No evidence-based slowing |
| Lapse (low coh) | 0.49 | ~0 | Punts half the time to relieve pressure |

Without the collapsing bound, PPO never learned to delay; it fires immediately and relies on random lapses to balance reward. Reward shaping alone is insufficient — structural inductive bias is required for realistic RTs.

### Hybrid DDM+LSTM WFPT curriculum (RDM) — `runs/hybrid_wfpt_curriculum/`

| Metric | Hybrid | Reference | Gap |
| --- | --- | --- | --- |
| RT intercept | 1.26 s | 0.76 s | 500 ms slower |
| RT slope | −767 ms/unit | −645 ms/unit | 19% overshoot |
| Psychometric slope | 7.33 | 17.6 | Too shallow |
| Bias | +0.001 | ≈0 | Matches |

Prioritizing WFPT loss early produced a strong negative chronometric slope — first evidence a mechanistic hybrid core is a viable path. RTs globally too slow; psych slope shallow. (See RT-ceiling note: this run's slope is partly ceiling-driven.)

### Hybrid with time-cost guardrails — `runs/hybrid_wfpt_curriculum_timecost/`

| Metric | Guardrailed | Reference | Gap |
| --- | --- | --- | --- |
| RT intercept | 0.883 s | 0.76 s | +123 ms |
| RT slope | −267 ms/unit | −645 ms/unit | ~40% of magnitude |
| Psychometric slope | 7.50 | 17.56 | Conservative |
| Bias | +0.24 | ≈0 | Small offset |
| History (win/lose/sticky) | 0.22 / 0.47 / 0.42 | 0.46 / 0.52 / 0.46 | Under-uses reward history |

Wider commit window (`max_commit_steps=180`) + heavier WFPT warmup prevents the 1.2 s ceiling collapse. History dipped below reference — the agent now under-perseverates.

### Soft RT penalty sweep

| Configuration | Chronometric slope | RT intercept | Notes |
| --- | --- | --- | --- |
| Time-cost guardrail (Attempt 1) | ≈0 ms/unit | 1.20 s | RT ceiling hit; collapsed to one side (bias ≈+6) |
| WFPT two-phase baseline | −505 ms/unit | 1.24 s | No RT penalty; healthy slope, high intercept |
| Soft RT (latest) | −165 ms/unit | 0.93 s | Soft penalty avoids ceiling; slope shallow, history under-shoots |

The soft penalty avoids the 1.2 s clamp but higher `rt_soft` weights drag the slope toward zero; history drops when RT pressure increases. Archived sweeps: `runs/archive/hybrid_wfpt_curriculum_timecost_soft_rt_attempt*/`.

### Cross-task observations and outstanding risks

- **RT realism requires policy-side latency.** Forcing an env delay merely shifts intercepts; agents need internal state/objectives that reward waiting.
- **Bias calibration is fragile.** Sticky-Q near-zero, PPO drifts heavily positive.
- **History kernels remain underfit** across both baselines.
- **Schema validation protects analysis** — NaN→null surfaces unstable regressions (e.g. PPO chrono fit fails because all RTs are identical).
- **Risks noted:** latency tuning is unprincipled; PPO absorbs penalty via random lapses; WFPT pipeline is delicate and slow on CPU; documentation drift (past READMEs overstated parity — this write-up replaces those claims).

---

## PPO Calibration Chronicle — Negative Results (Sept–Oct 2025)

A long series of PPO attempts that all failed to produce chronometric slopes — documented in full because they narrow the hypothesis space.

| Iteration | Key config | Intent | Outcome |
| --- | --- | --- | --- |
| `rdm_ppo_avgcost_v1` | Avg-reward `scale=0.5` | Penalize long trials | Froze on HOLD; flat curve; accuracy 48% |
| `rdm_ppo_avgcost_v2` | Avg-reward `scale=0.05` | Softer penalty | Accuracy ≈60%, RTs pegged at 60 ms, slope ≈0 |
| `rdm_ppo_calib_s005_*` | Avg-reward 0.05, urgency {0.8,1.2,1.6} | Policy-side urgency | RTs unchanged; high urgency destabilized (bias blow-up) |
| `rdm_ppo_calib_s002_*` | Avg-reward 0.02, urgency {0.8,1.2} | Reduce penalty | Slope still zero; psych slope oscillated |
| `rdm_ppo_confidence_v1` | Confidence reward + RT target | Reward waiting on evidence | Intercept 90 ms, slope stayed zero |
| `rdm_ppo_threshold_v1` | Evidence-threshold gate | Force bound crossing | RT stayed 60 ms (high coh hits instantly) |
| `rdm_ppo_threshold_v2` | Slower noise / higher gain | Harder accumulation | Psych overshot 6× reference; RT flat |
| `rdm_ppo_hold{20,30}_*` | Hold 20–30 + thresholds 1.5–3.0 | Motor-prep window | Intercepts 210–310 ms, no coherence effect |
| `rdm_ppo_thresh3_v3` | Threshold 3.0, no hold | Extreme gating | Reverted to immediate commits; accuracy ≈ chance |

**Lessons:** environment gating without stimulus dynamics fails; reward shaping alone cannot buy chronometry (bonuses only shift intercepts); urgency signals need conflict (with no cost to instant action they are ignored).

### October 11 sweep: stimulus pacing + time-cost probes

After adding coherence-dependent sampling and Tyro-exposed duration overrides:

| Run | Config diffs | Psychometric | Chronometric (ms) | History |
| --- | --- | --- | --- | --- |
| `rdm_ppo_coherence_long_20251011` | Stim 160, resp 200, coh-sampling, hold=10 | slope 3.35, bias ≈0 | flat 200 across | WS 0.51, LS 0.47, sticky 0.52 |
| `rdm_ppo_coherence_hold40_20251011` | Resp 280, hold=40, lower gain | slope 50.0, bias −0.50, lapse_hi 0.45 | flat 400 | WS 0.44, LS 0.67, sticky 0.37 |
| `rdm_ppo_coherence_soft_20251011` | Hold=5, thr 4.0, neg per-step reward | slope 6.0, bias −0.37, lapse_hi 0.21 | flat 250 | WS 0.65, LS 0.53, sticky 0.55 |
| `rdm_ppo_coherence_cost_20251011` | Neg per-step −5e-4, conf bonus 0.5 | slope 1.63, bias +0.17 | flat 100 | WS 0.44, LS 0.53, sticky 0.45 |
| `rdm_ppo_coherence_cost2_20251011` | Avg-reward 0.14, bound 3.5 | slope 6.77, bias −0.44, lapse_hi 0.16 | flat 150 | WS 0.76, LS 0.41 |

Time-cost micro-sweep (300k steps, 600-trial episodes):

| Run | Time pressure | Psychometric | Chronometric `rt_by_level` | Notes |
| --- | --- | --- | --- | --- |
| `rdm_ppo_chrono_A` | Avg reward 0.10, bound 3.0, urgency 0.8 | slope 18.1, bias −5.85, lapse_lo 0.16 | {0:20,.032:20,.064:30,.128:30,.256:25,.512:30} | WSLS lock-in (WS=1.0) |
| `rdm_ppo_chrono_B` | Avg reward 0.14, bound 2.5 | slope 3.69, bias ≈0 | all 10 ms | Penalty too strong—fires instantly |
| `rdm_ppo_chrono_C` | Time-cost controller (base 3e-4, growth 0.01) | slope 3.31, bias −0.03 | {…,.256:20,…30} | Slight dip at 0.256, far from macaque |

None produced the RT gradient. Avg-reward penalties require careful tuning (too low → camps on one choice; too high → instant commit). The time-cost controller yielded the only non-flat bin (20 ms at 0.256). **Pacing must emerge from evidence accumulation (gain/sigma schedules) plus calibrated time costs, not from hard response blocks or reward bribes.** The Nov 2025 plan of record: stimulus-pacing overhaul, average-reward-per-second objective with learnable non-decision time, history-informed baselines, and a documented calibration harness.

---

## Three Agent Archetypes and the Path Forward (Oct 2025)

Registry analysis (21 → 55 entries) revealed three architectural families, each capturing complementary aspects of animal behavior.

**Archetype A — Reward Optimizers (PPO):** `rdm_ppo_latest`, `20251017/18_ibl_ppo`. Psych slopes 50.0 (hyper-steep), chrono ≈0 (flat), highly variable bias (−207 to +0.52), moderate WS (0.55–0.78) as reward-exploitation artifact. Capture reward structure/accuracy ceiling; miss accumulation dynamics, speed-accuracy tradeoff, stable bias. Failure mode: without env shortcuts, fire instantly (RT=60 ms) or adopt pathological repetition (bias=−207, WS=1.0); `rdm_ppo_latest` reaches 49% lapse on low-coherence.

**Archetype B — History-Biased Heuristics (Sticky-Q):** `ibl_stickyq_latency`, `20251017/18_ibl_stickyq`. Psych 30.3–33.3, chrono ≈0 (flat), near-zero stable bias (−0.0001 to +0.005), strong history (WS 0.54–0.67, LS 0.38–0.60, sticky 0.52–0.67). Capture inter-trial dependencies; miss intra-trial dynamics. Tabular Q-learning with hand-engineered stickiness cannot learn to wait for evidence — no speed-accuracy mechanism.

**Archetype C — Mechanistic Integrators (Hybrid DDM+LSTM):** `hybrid_wfpt_curriculum`, `rdm_wfpt_regularized`, `20251017_rdm_hybridddml` + 13 variants. Psych 5.1–32.3 (variable), chrono −165 to −1828 ms/unit (consistently negative), bias near-zero when stable but prone to collapse (+6.06), history **consistently near chance** (WS 0.12–0.52). Capture the core decision process (best run −767 vs macaque −645); miss inter-trial memory — the LSTM "coach" sets DDM params at trial start but does not carry history forward. Failure modes: RT-ceiling collapse (`timecost_attempt1`: bias +6.06, sticky 0.998), history washout (`soft_rt`: WS→0.16), shallow psychometrics.

### The Decoupling Problem (as originally stated)

**We modeled the two key phenomena in isolation but not in one agent.** Intra-trial dynamics (chronometric slope) solved by Hybrid via WFPT curriculum; inter-trial dynamics (win-stay/lose-shift) solved by Sticky-Q via explicit history terms. Hybrid agents fail on history because the LSTM sets *static* DDM parameters at trial start with no dynamic pathway to bias accumulation, the parameters are frozen once set, and neither WFPT nor task reward explicitly penalizes failing to carry history.

### Architectural recommendations

- **Primary — Recurrent DDM (R-DDM):** compute instantaneous drift `drift_t = f(h_t, e_t, coherence)` so history exerts a *dynamic* bias overcome-able by strong evidence. Representational capacity to produce both negative chrono slopes and realistic history biases.
- **Incremental — History-aware Hybrid:** make history first-class by concatenating `prev_choice`, `prev_reward`, `prev_stimulus_strength` to the LSTM input, with an optional auxiliary WS/LS MSE loss. Flagged as a possible band-aid if the real issue is static parameter-setting.

### R-DDM infrastructure & Oct 2025 status

R-DDM gained end-to-end IBL + macaque support, task-aware rollouts, `--use-best` evaluation, and `rddm_sweep.py`. Despite the tooling, **R-DDM still fails to reproduce fingerprints**: best IBL psych ≈6 with flat chrono and saturated history; macaque plateaus at slope ≈0.1. The historic `20251019_rdm_hybridddml` (7-phase curriculum) remained the benchmark (psych ~32, chrono ~−1812); a three-phase sweep showed removing phases destroys the fingerprint (slopes 2–4, flat RT) — **keep the full multi-phase curriculum or modify in-place.** Registry reached 55 entries.

---

## Infrastructure and Loss Audits (Feb 2026)

### Per-trial history loss — the Decoupling fix hypothesis

Root-cause of why models learn chrono but fail history:

| Trainer | History mechanism | Gradient quality |
|---------|------------------|------------------|
| **R-DDM** | `_history_regulariser`: batch-mean MSE `(E[stay\|win] - target)²` | Differentiable but weak — O(1/N) per trial |
| **Hybrid** | `_estimate_history`: hard argmax on detached `prob_buffer` | **Zero gradient** (graph severed) |

Fix: `per_trial_history_loss()` in `agents/losses.py` — per-trial MSE `mean((stay_prob_i - target)²)` instead of batch-mean. By Jensen's inequality per-trial MSE ≥ batch-mean MSE, penalizing both mean deviation AND variance. Differentiable through `choice_prob`; convention-aware (R-DDM `no_action_value=-1`, Hybrid `no_action_value=0`). 12 tests in `tests/test_per_trial_history.py`. (WFPT normalization over-integration bug also **fixed** Feb 2026 — image charge positions corrected `z+2ka`→`a(z+2k)`; both series agree to 6 decimals. See audit below.)

### R-DDM formal evaluation — `runs/r_ddm_choice_only_v4`

| Metric | R-DDM (best ckpt) | R-DDM (regular) | IBL Reference | Notes |
| --- | --- | --- | --- | --- |
| Psychometric slope | 15.07 | 5.41 | 13.2 | Best-ckpt is closest match of any agent |
| Bias | 3.72 | 5.96 | +0.074 | |
| Win-stay | **0.953** | 0.865 | 0.73 | Extreme perseveration |
| Lose-shift | 0.071 | — | 0.34 | Nearly zero — ignores errors |
| Sticky choice | 0.939 | 0.919 | — | Pathological |
| RT (all levels) | 300.0 ms flat | — | 300 ms median | Motor delay only |
| p(right) | — | 0.14 | ~0.5 | |

Extreme stickiness (WS >0.95) far exceeds animal levels; despite balanced training-time WS (0.77), rollout locks onto one action. All RTs pinned at the 300 ms motor floor (choice-only, WFPT weight=0 → no RT signal). The best-ckpt psych slope (15.07) shows R-DDM can learn stimulus sensitivity, but perseveration + absent RT dynamics disqualify it as a behavioral model.

### WFPT implementation audit — `tests/test_wfpt.py` (20 tests)

1. **Drift convention inverted**: positive drift increases P(choice=0), opposite to standard DDM. Training compensates by learning inverted signs — end-to-end results unaffected.
2. **Density normalization degrades for strong parameters — FIXED**: small-time series had incorrect image charge positions (`z+2ka` instead of `a(z+2k)`), misscaling by `a²` when bound ≠ noise (∫density = 2.4 for drift=3, bound=2). Both series now agree to 6 decimals and integrate to 1.0.
3. **Edge cases handled**: near-zero drift, extreme biases (0.02, 0.98), tiny/huge RTs all give finite log-likelihoods; gradient flow verified for all 5 params.
4. **Symmetry preserved**: flipping drift sign mirrors likelihoods; zero drift gives equal likelihoods.

---

## The Decoupling Experiment and the Bias Artifact (Feb 2026)

A multi-phase test of whether per-trial history loss closes the Decoupling gap — culminating in the discovery that the "bias" and the "target" were both artifacts.

### Phases 1–3: per-trial history loss across scales

**Phase 1 (R-DDM, A–D):** varied per-trial weight {0.0, 0.5, 2.0} + combined. **All four produced identical rollout metrics** (WS 0.53, chrono 0.0, RT pinned 300 ms). R-DDM trains supervised on animal data and already achieves near-zero training loss (WS=0.73), so the per-trial loss has nothing to fix — the train→rollout gap (0.73→0.53) is distribution shift, not addressable by supervision. **Lesson: per-trial history loss can only help agents that train on their own rollouts.**

**Phase 2 (Hybrid, E–F, 10 episodes):** treatment improved every metric — chrono −19.3→−26.6 (+38%), RT range 110→155 (+41%), WS 0.176→0.194, prev_correct_beta −0.546→−0.124. But both runs shared ~83% leftward bias and ~90% right-lapse — a curriculum-level issue.

**Phase 3 (Hybrid, G–H, 30 episodes):** the effect **reversed** — control beat treatment on chrono (−24.2 vs −18.5), RT range (140 vs 120), prev_correct_beta (+0.36 vs +0.07).

| Metric | E:Ctrl(10) | F:Treat(10) | G:Ctrl(30) | H:Treat(30) | Animal |
| --- | --- | --- | --- | --- | --- |
| p(right) | 0.165 | 0.172 | 0.164 | 0.165 | ~0.5 |
| Psych slope | 6.8 | 7.2 | 6.8 | 6.7 | 10-20 |
| Chrono slope | −19.3 | −26.6 | −24.2 | −18.5 | −100+ |
| RT range (ms) | 110 | 155 | 140 | 120 | 200+ |
| Win-stay | 0.176 | 0.194 | 0.173 | 0.175 | 0.6+ |
| prev_correct_beta | **−0.546** | −0.124 | **+0.360** | +0.074 | >0 |

**Revised conclusions:** the E/F signal was **noise** (within the variance floor at ~2000 rollout trials). More training naturally solves prev_correct_beta (−0.546→+0.360 from 3× training, no per-trial loss) — the biggest finding. Per-trial loss may *interfere* at longer horizons. **The real bottleneck is the ~84% leftward bias (p(right)≈0.16)** which makes all other metrics unreliable — this must be fixed first (candidate fixes: stronger early choice-loss, anti-bias regularization, balanced stimulus presentation, curriculum gating on p(right) ∈ [0.4, 0.6]).

### Phase 4: the bias was never real — infrastructure discovery

Direct action counting revealed the bias was a **metric artifact**:

| Run | Left | Right | Hold | Commit Rate | p_right (all) | p_right (committed) |
| --- | --- | --- | --- | --- | --- | --- |
| G: Control | 2133 | 1973 | 7894 | 34.2% | 0.164 | **0.481** |
| H: Treatment | 2193 | 1979 | 7828 | 34.8% | 0.165 | **0.474** |
| I: Bias-fix ctrl | 2133 | 1952 | 7915 | 34.0% | 0.163 | **0.478** |
| J: Bias-fix treat | 2154 | 1945 | 7901 | 34.2% | 0.162 | **0.475** |

Among committed trials p_right ≈ 0.48 — nearly balanced. `p_right_overall = right/total` included holds; with 66% hold rate, `0.34 × 0.48 ≈ 0.16`. **Why 66% holds?** The DDM ran `max_commit_steps=200` but the env response phase was only 120 steps — planned commits at step 121–200 arrived after the env transitioned to outcome, logging `ACTION_HOLD`. The DDM *always* returns a committed choice; holds were purely a timing mismatch.

**Correctness fixes (not tuning):** `response_duration_override` matches window to planning horizon; `effective_max_commit = min(max_commit_steps, response_phase.duration_steps)`; `max_commit_steps` raised to 300; new metrics `p_right_committed` and `commit_rate`; `rt_ok` widened 2000→3500 ms. **All conclusions drawn from `p_right_overall ≈ 0.16` were invalid.**

### Phase 5: proper Decoupling — K2/L2

Re-run with bugs fixed (30 episodes, 400 trials/ep, seed 42, max_commit_steps=300):

| Metric | K2: Control | L2: Treatment | Animal Target |
| --- | --- | --- | --- |
| Commit rate | 1.000 | 1.000 | — |
| p_right_committed | 0.496 | 0.495 | ~0.5 |
| Psych slope | 10.7 | 10.6 | 10–20 |
| Psych bias | 0.007 | 0.005 | ≈0 |
| Chrono slope | −270 ms/unit | −264 ms/unit | −100 to −645 |
| RT range | 1350 ms | 1300 ms | 200–400 |
| Ceiling fraction | 0.17 | 0.17 | 0.0 |
| Win-stay | 0.486 | 0.498 | 0.6–0.8 |
| Lose-shift | 0.506 | 0.477 | 0.3–0.5 |
| Sticky choice | 0.488 | 0.504 | 0.5–0.7 |

**Psychometric and chronometric now qualitatively correct simultaneously** — psych 10.7 within range, chrono −270 with genuine slowing (890→2150 ms), 100% commit, 17% ceiling. **But history stays at chance** (WS≈0.49); per-trial loss (L2 vs K2) had no detectable effect. **The Decoupling Problem narrowed**: intra-trial dynamics reliably captured; the remaining gap is purely inter-trial. Architectural hypothesis: the DDM parameter space is too coarse (bias head ±0.02–0.10 vs bound 1.3; drift_gain ≈30 dominated by coherence). Candidate solutions floated: increase bias-head range, add a separate prior/lapse pathway, dynamic drift modulation (R-DDM), or an RPE signal bypassing the DDM bottleneck.

### Phase 6: history bias head — gradient isolation, three pathologies

A dedicated zero-init `history_bias_head` with gradient isolation from WFPT (9 unit tests). Six runs (weights 0.0–2.0, LR 3e-4–1e-2, scales 0.5–1.0, ±freeze, three formulations) all preserved K2 intra-trial performance but **none moved history above chance** (WS 0.485–0.502). Three stacked gradient pathologies:

1. **DDM choice-prob saturation** — gradient zeros at high coherence:

| Coherence | 0.000 | 0.032 | 0.064 | 0.128 | 0.256 | 0.512 |
| --- | --- | --- | --- | --- | --- | --- |
| dp/d(hb_weight) | 0.210 | 0.160 | 0.076 | 0.010 | 0.000087 | 0.000000 |

Fix (v5): sigmoid for the training-loss path (all coherences give gradient 0.4–5.1); rollout still uses the real DDM.
2. **Left/right gradient cancellation** — P(right) parameterization anti-aligns win-trial gradients (cosine −0.48, **72% cancellation**). Fix (v6): re-parameterize as P(stay) → cancellation drops to 56% (cosine +0.27).
3. **Optimization instability** — session-dependent LSTM hidden states push weights in different directions (weights oscillate 0.01→0.0002→0.004→0.003). The LSTM *does* encode history (mean abs diff 0.194 between post-win/post-loss states; drift_gain differs by 5.7) but a small linear head cannot extract it from the 64-dim space in budget.

**Conclusions:** gradient isolation works; the LSTM encodes history non-linearly-extractably; the DDM bottleneck is real; it is not a loss-weighting issue; an **architectural change is needed** — a separate history input pathway, aligning with neuroscience (PFC/basal-ganglia history vs parietal accumulation). Chosen: **Option 1, Separate History Stream** — a small MLP taking (prev_action, prev_reward) directly, outputting a "stay tendency" that shifts the DDM starting point, additive and never touching what works. Rejected: Dynamic R-DDM (risks intra-trial performance) and prior-mixture (a post-hoc statistical correction, not mechanistic). Biological mapping: LIP→LSTM/DDM (accumulation), PFC+BG→history network, PFC→LIP baseline shift→starting point, dopamine→history-loss gradient.

### Phase 7: the reference-data discovery — the target didn't exist

The separate history network (v7) was architecturally sound (output max 0.096 vs 0.003 for the LSTM head; converges in ~20 steps in isolated tests), but rollout history stayed at chance (WS=0.492, LS=0.517). The reason:

| Metric | Macaque reference (aggregate) | Per-session range | Our model |
| --- | --- | --- | --- |
| Win-stay | **0.458** | 0.222–0.548 | 0.492 |
| Lose-shift | 0.515 | 0.000–0.739 | 0.517 |

**The Roitman & Shadlen macaque does not show above-chance win-stay** (aggregate 0.458; only 7/27 sessions > 0.5) — consistent with an overtrained monkey ignoring outcomes. **Our model at 0.492 already matched the reference better than any WS>0.5 target would.** Six experiments and three architectural fixes were spent optimizing toward a target that doesn't exist in the data. **The methodological failure: we never checked whether the reference exhibited the effects we were chasing.** What was actually learned: the three gradient pathologies are real and reusable; the separate history stream is the correct approach but **untested against data with genuine history effects**; the Decoupling Problem as stated was partly an artifact for this dataset (both dynamics correctly captured); less-trained-animal data would provide the missing training signal (a testable overtraining-gradient prediction). Next: switch to IBL mouse (WS 0.73, LS 0.34).

---

## Solving the Decoupling Problem: Drift-Rate Bias, Attention Gating, Differentiable DDM (Feb 2026)

### Phase 8: IBL adaptation — drift-rate bias solves the Decoupling Problem

The Hybrid was parameterized by task (`--task ibl_2afc`), not forked — identical architecture/loss/history_network, only data loading and rollout differ (signed contrast [-1,1], IBL RT targets, block-prior-aware zero-contrast logic, extended response window).

**Starting-point bias (v1–v3)** plateaued at WS ≈0.544 regardless of duration/weight, even though the trained network output the correct sigmoid P(stay)=0.725 after wins:

| Run | history_bias_scale | Win-stay | Lose-shift | Chrono slope |
| --- | --- | --- | --- | --- |
| v1 | 0.5 | 0.547 | 0.525 | -73.1 |
| v2 (long) | 0.5 | 0.544 | 0.530 | -77.0 |
| v3 (strong) | 1.0 | 0.544 | 0.533 | -77.3 |
| IBL reference | — | **0.724** | **0.427** | negative |

**Root cause:** starting-point bias only affects ambiguous trials:

| \|Contrast\| | 0.0 | 0.0625 | 0.125 | 0.25 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| P(right) shift | +11.5% | +5.1% | +1.2% | +0.1% | +0.0% |

At \|contrast\| ≥ 0.125 the drift rate overwhelms the shift; ~60% of trials qualify, so starting-point can move overall P(stay) only ~2–4 points. **Real mice show history effects even on easy trials — history must affect accumulation itself.**

**The fix — drift-rate bias** (`history_drift = stay_tendency * history_drift_scale * prev_direction`, added to `effective_drift`), affecting ALL trials:

| Run | history_drift_scale | Win-stay | Lose-shift | Sticky | Psych | Chrono |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (no drift) | 0.0 | 0.547 | 0.525 | 0.530 | 6.70 | -73.1 |
| v4 | 5.0 | 0.585 | 0.487 | 0.569 | 7.28 | -72.2 |
| v5 | 8.0 | 0.607 | 0.458 | 0.592 | 6.61 | -72.5 |
| **v6 (max)** | **15.0** | **0.655** | **0.402** | **0.642** | 6.04 | -66.6 |
| IBL reference | — | **0.724** | **0.427** | **0.692** | ~13.2 | negative |

For the first time in 60+ experiments, an agent produced all three fingerprints (negative chrono, above-chance history, psychometric discrimination) via the **Attention-Gated History Bias** mechanism. History requires drift-rate bias, not just starting-point bias. A WS/discrimination tradeoff appears (psych 6.70→6.04 as drift rises) — mirroring mice with stronger history biases being slightly less accurate.

### Phase 9: the joint-learning reality check + attention gate

v1–v6 secretly **hardcoded** `history_drift_scale=15.0` and `history_bias_scale=1.0`, injected at rollout while the history network trained on a disjoint cross-entropy objective — the WFPT gradient never flowed into the history network. Making them true learnable `nn.Parameter`s and unifying the graph, a 5-seed sweep (`runs/seed_sweep_v6_joint`, v6_max config) revealed **mode collapse**:

| Seed | Win-stay | Lose-shift | Sticky | Psych slope |
| --- | --- | --- | --- | --- |
| 42 | 0.8792 | 0.1382 | 0.8737 | 2.6339 |
| 123 | 0.9138 | 0.1036 | 0.9078 | 2.0918 |
| 256 | 0.9228 | 0.1040 | 0.9136 | 2.0488 |
| 789 | 0.9133 | 0.1209 | 0.9020 | 2.1266 |
| 1337 | 0.8985 | 0.1443 | 0.8850 | 2.4386 |
| **Mean** | **0.9055** | **0.1222** | **0.8964** | **2.2679** |
| Target | 0.7240 | 0.4270 | — | 13.2000 |

All seeds learned to **ignore the stimulus** (psych crashed to ~2.2) and hit the same button (WS ~0.90). The claim that the agent can disentangle evidence from history without regularization is **officially false**.

**Attention-Gated History Bias fix** — define confidence `min(|stimulus|,1.0)`, scale history drift by `(1 - confidence)`:

| Seed | Win-stay | Lose-shift | Sticky | Psych slope |
| --- | --- | --- | --- | --- |
| 42 | 0.8174 | 0.2253 | 0.8063 | 4.2592 |
| 123 | 0.8489 | 0.1822 | 0.8401 | 3.7434 |
| 256 | 0.8491 | 0.1871 | 0.8393 | 3.6543 |
| 789 | 0.8478 | 0.1930 | 0.8366 | 3.7164 |
| 1337 | 0.8343 | 0.2408 | 0.8149 | 4.1748 |
| **Mean** | **0.8395** | **0.2057** | **0.8274** | **3.9096** |
| Target | 0.7240 | 0.4270 | — | 13.2000 |

The gate shatters mode collapse across all seeds (psych rebounds 2.26→3.91, WS 0.90→0.84). Still shallower than target 13.2, but the agent can now balance history and evidence dynamically. Interpretation: the Decoupling Problem is a feature of unconstrained joint learning; biology uses top-down attentional gating (`gated_history_drift = history_drift * (1.0 - confidence)`) as a structural guardrail; artificial curriculum tricks (freeze-then-train) are biologically implausible whereas the gate enables stable continuous joint learning. Recommended future mechanisms flagged: sensory-evidence obligation (structural bounds), asymmetric/attention-gated history, and RPE separation.

### Phase 10: differentiable DDM simulator + psychometric-slope breakthrough

**The `tanh(κ)` exploit:** under high choice-loss pressure using `E[RT] = (A/v)·tanh(v·A/σ²)`, the LSTM pushed bound `A→∞` and drift `v→0`, collapsing `tanh→0` and the RT gradient — every trial timed out (100% degenerate). **Fix: differentiable Euler-Maruyama simulation** (`ΔE = v·Δt + σ·√Δt·N(0,1)`): evidence trajectory via cumsum over 120 steps, soft boundary `σ((evidence−bound)/temp)`, commit density via cumulative product, `E[RT] = Σ(t·commit_density)·step_ms + non_decision_ms`, timeout penalty `P(timeout)·max_steps·10`. The agent can no longer hide behind the asymptote. The 3-phase curriculum (RT-only → add-choice → full-balance) **removes WFPT entirely** and uses drift-magnitude regularization.

3×3 sweep `drift_scale ∈ {10,20,30}` × `choice_weight ∈ {0.5,1.0,1.5}`:

| drift | choice_w | Psych | Chrono | WS | LS | Bias | Quality |
|-------|----------|-------|--------|----|----|------|---------|
| 10 | 0.5 | 5.25 | ~0 | 0.718 | 0.167 | -0.155 | **degenerate** |
| 10 | 1.0 | **14.17** | -288 | 0.657 | 0.196 | -0.026 | all pass |
| 10 | 1.5 | **14.40** | -285 | 0.659 | 0.188 | -0.026 | all pass |
| **20** | **0.5** | **13.78** | **-286** | **0.654** | **0.211** | **-0.023** | **all pass** |
| 20 | 1.0 | 22.25 | -67 | 0.554 | 0.553 | 0.002 | all pass |
| 20 | 1.5 | 23.73 | -66 | 0.548 | 0.551 | 0.002 | all pass |
| 30 | 0.5 | 23.32 | -65 | 0.545 | 0.573 | 0.001 | all pass |
| 30 | 1.0 | 22.05 | -63 | 0.550 | 0.563 | 0.004 | all pass |
| 30 | 1.5 | 24.08 | -65 | 0.550 | 0.564 | 0.001 | all pass |

Three regimes: **degenerate** (insufficient choice pressure — simulator fix necessary but not sufficient); **sweet spot** (psych 13.8–14.4); **over-discriminating** (psych 22–24, history collapses to chance). The **history-accuracy tradeoff** quantified: psych 5–6 → WS 0.66–0.72; psych 13–15 → WS 0.65–0.66; psych 22–24 → WS 0.54–0.55. The three-regime structure is a testable prediction across individual IBL mice.

**Multi-seed validation — the single-seed 13.78 did NOT replicate.** It was an artifact of RT-ceiling saturation (28% lapse, 2/6 RT levels at 3000 ms, step-function RT `3000/3000/2615/1180/580/340`), later fixed by trainer refactoring:

| Metric | Post-refactor 5-seed mean ± SD | IBL Target | Verdict |
|--------|-----------|------------|---------|
| Psych slope | 22.96 ± 1.94 | 13.2 | 74% overshoot |
| Chrono slope | -64.1 ± 2.1 | -36 | 1.78× overshoot |
| Win-stay | 0.565 ± 0.009 | 0.724 | Below |
| Lose-shift | 0.551 ± 0.030 | 0.427 | Above |
| Lapse | ~0.002 | ~0.05 | Too accurate |
| Bias | -0.004 ± 0.010 | ~0 | Excellent |

RT now smooth (`910/850/670/480/330/250`, range 660 ms) — healthier despite the psych overshoot. The apparent 96% match was a coincidence of lapse + ceiling. (Old-arch drift sweep {10,12,14} confirmed psych ~21 insensitive to `drift_scale`.)

**Asymmetric history networks + stochastic lapse** added: separate `win_history_network`/`lose_history_network` routed by `prev_reward > 0.5`; a rollout attention-lapse gate.

- **Learnable lapse (negative result):** a learnable `lapse_logit` looked promising at drift=20 (psych 5.60, lapse_lo 0.042, LS 0.414) but the optimizer **exploited** it — at drift {25,30,35} lapse tripled to ~15% and turned symmetric (guessing 50/50 lowers choice loss on hard trials). Animal lapse is a hardware property, not a strategy. **Decision: fixed `lapse_rate=0.05` as a rollout-only Bernoulli gate.**
- **Training lapse double-counting (negative result):** blending fixed lapse into training prob (`(1-lapse)·prob + lapse·0.5`) stuck psych at ~8.5 regardless of drift {20,22,25} — reference data already contains the animal's lapse. **Lapse belongs in rollout only.**
- **Rollout-only lapse sweep:** still too shallow (psych ~9.5) and insensitive to drift — leading to the curriculum confound.

**The curriculum confound (critical discovery):** the new-architecture sweeps used the 7-phase WFPT curriculum; the old psych=22.96 run used the 3-phase. A controlled swap isolated it:

| Configuration | Psych | Chrono | WS | LS | Lapse |
|---------------|-------|--------|----|----|-------|
| Old arch + 3-phase | 22.96 ± 1.94 | -64.1 ± 2.1 | 0.565 | 0.551 | ~0.002 |
| **New arch + 3-phase** | **18.34 ± 2.12** | **-63.5 ± 5.1** | **0.556** | **0.553** | **~0.019** |
| New arch + 7-phase | 9.57 ± 0.37 | -40.7 ± 1.8 | 0.566 | 0.516 | ~0.026 |
| **Animal target** | **13.2** | **negative** | **0.724** | **0.427** | **~0.05** |

**The 7-phase WFPT curriculum caused the psych collapse (23→9.5), not the architecture.** WFPT warmup (15 epochs, choice loss off) lets the optimizer settle into a high-noise, low-sensitivity regime. The 3-phase curriculum teaches RT first via MSE, then layers choice on a stable foundation — mirroring sensory circuits maturing before decision circuits. **Always use the 3-phase (`--no-use-default-curriculum`).**

**Drift calibration — the dead-knob discovery:**
- **v1:** `drift_scale` {12,14,16,18} gave identical psych ~21.5 — it only controls drift_head *initialization*; the `drift_magnitude` loss pulls drift_gain to a hardcoded target 12.0. **`drift_scale` is a dead knob.**
- **v2:** making `drift_magnitude_target` configurable gives a clean monotonic lever:

| drift_magnitude_target | Psych | Chrono | WS | LS |
|------------------------|-------|--------|----|----|
| **6.0** | **12.76 ± 1.04** | **-64.1 ± 2.4** | **0.556 ± 0.005** | **0.543 ± 0.004** |
| 7.0 | 15.19 ± 0.82 | -67.3 ± 3.2 | 0.557 | 0.560 |
| 8.0 | 15.96 ± 1.41 | -66.7 ± 3.6 | 0.563 | 0.549 |
| 9.0 | 17.08 ± 1.22 | -69.3 ± 2.6 | 0.557 | 0.556 |
| 12.0 (old default) | ~21.5 ± 1.5 | ~-70 ± 4.0 | ~0.556 | ~0.555 |
| **IBL target** | **13.2** | **-36 ms/unit** | **0.724** | **0.427** |

**Target=6.0 gives psych 12.76**, bracketing 13.2. Joint-production test passed — calibrating sensitivity does NOT break other fingerprints (chrono, WS, LS unchanged; 100% commit). `drift_magnitude_target` is analogous to standard DDM fitting: one parameter fit to the psychometric curve, the rest must emerge architecturally.

### Phase 15: history finetuning — per-trial sigmoid proxy loss is dead

Five infrastructure bugs found & fixed: config→model passthrough missing (`history_bias_scale`/`history_drift_scale` used constructor defaults 0.5/0.0, ignoring config 2.0/0.3); rollout read config not model params; loss/rollout scale mismatch (`sigmoid(tendency·scale·4.0)` vs `tendency·scale·bound≈1.0`); sigmoid saturation (range [0.378,0.622] cannot reach WS 0.724, and the optimizer shrank scale 0.5→0.216); incomplete freeze exclusion. After all fixes, 4 controlled runs (seed 42):

| Run | choice_w | per_trial_w | hb_scale | hd_scale | WS | LS | Psych |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pre-fix baseline | 0.5 | 0.5 | 0.5 (not passed) | 0.0 (not passed) | 0.556 | 0.513 | 13.75 |
| Post-fix v1 | 0.5 | 1.0 | 0.5 (not passed) | 0.3 (not passed) | 0.549 | 0.553 | 12.85 |
| v3 (passthrough) | 0.5 | 1.0 | 2.0 | 0.3 | 0.522 | 0.572 | 13.66 |
| v4 (no choice loss) | 0.0 | 1.0 | 2.0 | 0.3 | 0.516 | 0.565 | 12.94 |
| **IBL target** | — | — | — | — | **0.724** | **0.427** | **13.2** |

Even with choice loss removed, WS **decreased** (0.556→0.516). **Root cause:** the loss trains `stay_tendency` through a **sigmoid proxy** (`P(stay)=sigmoid(tendency·scale·bound)`), but behavioral WS/LS emerge from the **DDM simulation** — completely different transfer functions. Like `history_supervision` (dead because detached), the `per_trial_history` loss converges while behavior does not change. Psych stayed stable (12.85–13.75) — the Phase 4 freeze mechanism works. Future approaches: backprop through the DDM directly, RL with shaped reward, or direct parameter sweep of hb/hd scales.

---

## The prev_reward Bug and Co-Evolution Training (Feb 2026)

### Phase 16: the rollout prev_reward bug

**`prev_reward` was always 0.0 in every rollout ever run.** The outcome-phase check used `info["phase_step"] == 0`, matching the transition step before reward delivery; the real reward arrived at `phase_step == 1`. The env computes reward *before* advancing `phase_step` but returns `info` *after*. **Fix: `== 0` → `== 1`.**

**Impact:** every trial routed through `lose_history_network`; `win_history_network` output was never used in any rollout. This explains why WS was stuck at ~0.556 despite every intervention, why the Phase 16a scale sweep (16 combos) produced identical WS/LS, and why Phase 15 per-trial experiments failed (training networks whose outputs were never used).

- **16a (pre-fix, negative):** `history_bias_scale × history_drift_scale` (hb {1,2,4,8} × hd {0,0.3,1,2}, 16 combos) all gave WS ~0.55–0.59, LS ~0.52–0.57; higher scales → *smaller* output weights.
- **16b (pre-fix injection diagnostic):** `win_tendency` had ZERO effect (all trials routed through lose pathway); only `lose_tendency` moved metrics.
- **16c (post-fix injection diagnostic):** `win_tendency` now has clear monotonic effect:

| win_t | lose_t | WS | LS | Psych | Chrono |
|-------|--------|----|----|-------|--------|
| 0.0 | 0.0 | 0.539 | 0.539 | 12.14 | -62.6 |
| 0.5 | 0.0 | **0.727** | 0.516 | **13.5** | **-34.6** |
| 0.5 | 0.1 | 0.764 | **0.449** | **13.6** | -29.1 |
| 0.8 | 0.0 | 0.736 | 0.505 | **14.1** | -31.9 |
| Target | | 0.724 | 0.427 | 13.2 | -36 |

Key findings: the DDM **can** express correct WS/LS (0.727 ≈ target at win=0.5); optimal ≈ win_t 0.5, lose_t 0.05; **history bias IMPROVES psych slope** (12.14→13.5); **chrono self-calibrates** (−34.6 near target vs baseline −64); **all four metrics simultaneously matchable** — previously thought architecturally impossible.

### Phase 17: co-evolution training — simultaneous calibration

Post-fix training from scratch without injection (17a, 3 seeds) improved WS only 0.556→0.578 — the fix affects rollout routing, not what the supervised networks learn. Solution: **co-evolution** — train from scratch with fixed injection active so the LSTM/DDM heads co-evolve alongside history.

**17b injection fine sweep — the fundamental WS/psych tradeoff:**

| win_t | lose_t | WS | LS | Psych | Chrono |
|-------|--------|----|----|-------|--------|
| 0.15 | 0.00 | 0.625 | 0.536 | 13.07 | -58.9 |
| 0.20 | 0.00 | 0.663 | 0.517 | 12.18 | -54.2 |
| 0.25 | 0.00 | 0.691 | 0.509 | 11.32 | -48.3 |
| **0.30** | **0.00** | **0.724** | **0.500** | **10.30** | **-42.5** |
| 0.35 | 0.00 | 0.753 | 0.492 | 9.42 | -36.9 |

At win_t=0.30, WS=0.724 (exact) but psych drops to 10.3. Starting-point bias shifts the accumulator, degrading sensitivity at all contrasts because the gate `(1-|stimulus|)` only fully closes at |stimulus|=1.0. A scale-ratio sweep confirmed drift-rate bias degrades psych equally (same pathway; `effective_history_bias_scale` has a `clamp(min=1.0)` floor). **The pretrained evidence circuits cannot compensate post-hoc — they must co-evolve.**

**17c co-evolution v1** (win=0.3, lose=0.08, target=6.0, 3 seeds): chrono −33.1 ± 3.8 (within 8% of −36, was −64), WS 0.714 ± 0.005 (was 0.556), lapse 0.056 — but psych dropped to 10.11 ± 0.52 (evidence circuits need stronger drift).

**17d drift recalibration** (fixed injection win=0.3 lose=0.12):

| Target | Psych | Chrono | WS | LS | Lapse |
|--------|-------|--------|----|----|-------|
| 7 | 10.37 | -32.5 | 0.710 | 0.480 | 0.060 |
| 8 | 11.58 | -33.9 | 0.705 | 0.477 | 0.056 |
| **9** | **12.85** | **-36.5** | **0.703** | **0.480** | **0.062** |
| 10 | 13.44 | -32.7 | 0.709 | 0.477 | 0.085 |

**Target=9 is the sweet spot** (psych 12.85, chrono −36.5 essentially exact).

**17e win_t × lose_t fine-tuning** (target=9 locked; v3 sweep, v4 12-combo confirmed the same tradeoff):

| win_t | lose_t | WS | LS | Psych | Chrono | Lapse |
|-------|--------|----|----|-------|--------|-------|
| 0.30 | 0.12 | 0.704 | 0.466 | 13.08 | -31.0 | 0.074 |
| **0.30** | **0.15** | **0.704** | **0.449** | **13.16** | **-34.6** | **0.075** |
| 0.30 | 0.18 | 0.700 | 0.433 | 13.00 | -32.3 | 0.066 |
| 0.32 | 0.15 | 0.719 | 0.457 | 12.44 | -34.6 | 0.075 |
| **0.32** | **0.17** | **0.724** | **0.441** | **11.44** | **-30.9** | **0.069** |
| 0.35 | 0.18 | 0.740 | 0.436 | 10.28 | -31.7 | 0.070 |

A persistent WS/psych tradeoff: exact WS=0.724 requires win_t≥0.32, degrading psych below 12. Two candidates emerge — **A (psych-optimized, win=0.30/lose=0.15)** and **B (WS-optimized, win=0.32/lose=0.17, single seed: psych 11.44, chrono -30.9, WS 0.724, LS 0.441, lapse 0.069)**.

**5-seed validation of Candidate A:**

| Seed | Psych | Chrono | Win-stay | Lose-shift |
|------|-------|--------|----------|------------|
| 42 | 13.16 | -34.6 | 0.704 | 0.449 |
| 123 | 11.79 | -34.9 | 0.705 | 0.451 |
| 456 | 11.96 | -36.1 | 0.709 | 0.462 |
| 789 | 11.99 | -32.0 | 0.713 | 0.459 |
| 1337 | 12.99 | -33.5 | 0.697 | 0.465 |
| **Mean ± std** | **12.38 ± 0.64** | **-34.2 ± 1.8** | **0.706 ± 0.008** | **0.457 ± 0.007** |

Progress vs the old single-session targets (superseded; see Methodological Notes for corrected per-session targets): Candidate A gives psych 12.38 ± 0.64, chrono −34.2 ± 1.8, WS 0.706 ± 0.008, LS 0.457 ± 0.007, lapse 0.075, bias 0.000. History effects, chrono, LS, and lapse fall within the reference per-session range; psych sits below the per-session mean 20.0 (the old 13.2 single-session target understated this gap). Progress summary vs Phase 15 start: psych 12.76→12.38 (within 1.4 std of reference mean); chrono −64.1→−34.2 (corrected from overshoot); WS 0.556→0.706 (within range); LS 0.543→0.457 (within range); lapse ~0.025→0.075 (within range).

**What was proved:** co-evolution is necessary (target=6 no-history + injection → psych 10.1; co-evolution at target=9 → 12.38); the architecture produces all six fingerprints simultaneously; a fundamental WS/psych tradeoff exists (architectural ceiling from the attention gate); drift target must be re-calibrated with history (6→9); chrono self-calibrates with history (−64→−34.2, not independently tuned — history compresses RT variation, biased trials are faster). **What remains:** history effects are **injected, not learned** (networks still output near-zero — needs DDM-direct backprop or RL); the WS/psych tradeoff is a ceiling (a sharper gate `(1-|stimulus|)^k` might decouple it); psych below the per-session reference mean; chrono target uncertainty high (per-session −2 to −202, std 64).

---

## Plastic History and the Adaptive-Control Reframing (March 2026)

Can the agent learn history from a biologically plausible local rule (RPE + eligibility traces, dopamine-like) rather than injected tendencies?

| Variant | Approach | Result |
|---------|----------|--------|
| 1. Assisted (`plastic_history_seed_*`) | Kept injection scaffolding + online plastic fast weights + critic | Unstable, worse than injection baseline |
| 2. Pure (`plastic_history_pure_seed_*`) | Removed injection/distillation, online plasticity only | 5-seed: psych 16.91 ± 1.60, chrono -59.08 ± 1.47, WS 0.572 ± 0.017, LS 0.521 ± 0.020 — stable, promising, but WS below injection baseline |
| 3. Pure v2 | Longer/stronger phase-4 finetune | Stable but not better; WS fell to 0.545 |
| 4. Pure v3 | Increased plastic gain | Plastic pathway active (mean_plastic_stay_tendency 0.004→0.10–0.21) but WS did not improve — bottleneck isn't "too little plasticity" |
| 5. Pure v4 | Separate +/− plasticity + counterfactual switch after loss | Pathway strongly active (~0.22–0.29) but WS=0.540, LS=0.502 — produces generic bias, not WS/LS asymmetry |
| 6. Pure v5 | Loss-side reinterpreted as explicit lose-shift pressure | Effectively identical to v4 — more coherent story, no optimization change |

(Pure 5-seed reference row: IBL per-session mean psych 20.0 ± 5.7, chrono -51 ± 64, WS 0.72 ± 0.08, LS 0.47 ± 0.10.)

**Learned:** teacher guidance is not learned history (didn't survive removal); local plasticity is necessary but not sufficient; the current history-head family is too narrow (learns recent-choice bias, not persist/switch/explore under uncertainty); reward and punishment should not be a single signed stay scalar.

**New conclusion — reframe the question.** The plausible mammalian story is a control system: **evidence circuit** (what does the stimulus say?) + **value/critic** (better or worse than expected?) + **persistence controller** (retry despite failure when evidence was weak?) + **exploration controller** (sample alternatives when the world may have changed?) + **arbitration** (combine without overwriting strong sensory evidence). This moves the question from "can it learn the right history scalar?" to "can it learn the control system that produces animal-like persistence, switching, and exploration?" Keep the co-evolution result as the best validated behavioral match; prototype a new agent family; evaluate new metrics beyond WS/LS (retry-after-failure on ambiguous trials, exploratory switching, persistence-as-function-of-confidence).

---

## Adaptive Control Experiments (May 2026)

A separate agent path (`agents/adaptive_control_*`, `scripts/train_adaptive_control.py`) leaving the validated hybrid path intact. Biological framing is a deliberate analogy, not anatomy.

### Implementation guardrails (scientifically important)

1. Clean no-control lesion must zero **all** adaptive outputs, not one head.
2. Explicit outcome valence — failure teaching can't depend only on critic PE (a calibrated critic silences failure updates).
3. Bounded residual overlays on the evidence core, not a replacement.
4. Evidence preservation — residuals regularized more on high-evidence trials.
5. Long IBL response window — the env's short default produced an artificial 300 ms RT ceiling.
6. Nonlinear uncertainty gate (`control_uncertainty_power=2.0`).

### Phase-1 calibrated defaults & 5-seed validation

Default profile: **`persistence_only`** (control + persistence on, exploration off). Defaults: `drift_scale=6.0`, `persistence_bias_scale=1.6`, `control_uncertainty_power=2.0`, `control_residual_limit=0.35`, `control_pressure_limit=0.35`.

Run `runs/adaptive_control_validation_suite_phase1/`:

| Condition | Psych | Chrono | Retry gap | RT ceiling | Degenerate |
|-----------|-------|--------|-----------|------------|------------|
| true no-control | 27.71 ± 3.28 | -48.54 ± 7.05 | 0.057 ± 0.062 | 0/5 | 0/5 |
| persistence-only | 21.75 ± 2.69 | -33.47 ± 4.49 | 0.164 ± 0.108 | 1/5 | 0/5 |
| full control | 22.26 ± 1.80 | -33.97 ± 4.02 | 0.165 ± 0.045 | 0/5 | 0/5 |

Paired deltas vs no-control: full control +0.109 ± 0.086 retry (5/5 seeds; Δpsych -5.45 ± 3.79, Δchrono +14.57 ± 8.96); persistence-only +0.107 ± 0.136 (3/5; Δpsych -5.96 ± 2.22, Δchrono +15.07 ± 7.01). **The clean claim is persistence/adaptive retry.** Full control gives the most consistent lift but bundles exploration; persistence-only recovers nearly the same mean (0.164 vs 0.165) with exploration off — the **recommended default**.

### Gate lesion (`..._phase1_gate/`)

The linear-gate lesion (`control_uncertainty_power=1.0`) did not fail catastrophically: nonlinear full control retry +0.109 (5/5), linear-gate +0.087 (3/5). The nonlinear gate is not strictly necessary; sharpened gating makes the retry effect **stronger and more reliable**. The mechanism is uncertainty-gated adaptive control, not the exact exponent.

### Exploration follow-up (`..._phase1_exploration/`) — negative result

Added `exploration_only` and stale-state probes (`switch_after_streak_weak`, `switch_after_fresh_weak`, `stale_switch_lift_weak = streak − fresh`):

| Condition | Psych | Chrono | Retry gap | Stale-switch lift | Degenerate |
|-----------|-------|--------|-----------|-------------------|------------|
| true no-control | 27.71 | -48.54 | 0.057 | -0.073 | 0/5 |
| exploration-only | 24.00 | -38.83 | 0.092 | -0.160 | 0/5 |
| persistence-only | 21.75 | -33.47 | 0.164 | -0.159 | 0/5 |
| full control | 22.26 | -33.97 | 0.165 | -0.152 | 0/5 |

Paired vs no-control: retry positive in 3/5 (exploration), 3/5 (persistence), 5/5 (full); stale-switch lift 0/5 positive in every condition. **Valuable negative result:** the weak-failure retry signature holds, but **all conditions made stale-switch lift MORE negative** — after repeated rewarded choices the agent becomes more *exploitative*, not exploratory. Exploration is not independently validated and stays out of the default profile.

### Unrewarded/volatility probe screen (`..._exploration_probe_5seed/`)

Added `unrewarded_switch_lift_weak` and `volatile_switch_lift_weak` (light budget, episodes=3, epochs=1; 0/5 degenerate, 0/5 ceiling):

| Condition | Psych | Chrono | Retry gap | Unrewarded-switch lift | Volatile-switch lift |
|-----------|-------|--------|-----------|------------------------|----------------------|
| true no-control | 30.61 | -81.92 | 0.039 | 0.094 | -0.002 |
| exploration-only | 29.85 | -61.93 | 0.077 | 0.131 | 0.052 |
| persistence-only | 30.27 | -61.14 | 0.121 | -0.049 | 0.037 |
| full control | 29.66 | -60.87 | 0.121 | 0.086 | 0.069 |

Paired vs no-control: retry +0.038 (exploration, 3/5), +0.082 (persistence, 5/5), +0.082 (full, 5/5); unrewarded-switch +0.037 (explore, 4/5), −0.143 (persist, 2/5), −0.008 (full, 4/5); volatile-switch +0.054 (explore, 3/5), +0.040 (persist, 3/5), +0.071 (full, 4/5). Retry cleaner than exploration. The unrewarded probe's repeated-failure counts are too small to carry a claim. **Local-volatility is the better future readout** (hundreds of matched events, strongest in full control), but persistence-only also shows positive volatility lift. Next claim-bearing step: a sharper volatility-specific lesion/gate, or a PRL/DMS transfer where exploration is task-relevant.

### Hidden block-switch bridge metric

`block_switch_probe` detects `block_prior` reversals; `adaptation_lift = new-prior choice rate (trials 6-10) − (trials 1-5)`:

| Condition | Block switches | Early | Late | Adaptation lift | Zero-contrast new-prior |
|-----------|----------------|-------|------|-----------------|-------------------------|
| IBL reference | 137 | 0.585 | 0.747 | +0.162 | 0.567 |
| true no-control | 30 | 0.692 | 0.692 | -0.000 | 0.383 |
| exploration-only | 30 | 0.640 | 0.720 | +0.080 | 0.435 |
| persistence-only | 30 | 0.592 | 0.648 | +0.056 | 0.383 |
| full-control | 30 | 0.612 | 0.680 | +0.068 | 0.396 |

A bridge metric, not a success claim: reference animals show slow adaptation; adaptive conditions introduce some delayed adaptation but smaller than reference and not cleanly exploration-specific.

### Block-switch-focused matched suite (`..._block_switch_focus_v1/`)

5 paired seeds, 6 episodes, 800 trials/ep, 30 biased switches/run:

| Condition | Psych | Chrono | Retry gap | Block-switch lift | Early | Late |
|-----------|-------|--------|-----------|-------------------|-------|------|
| true no-control | 34.30 | -80.01 | 0.068 | +0.033 | 0.711 | 0.744 |
| exploration-only | 28.46 | -56.86 | 0.031 | +0.136 | 0.655 | 0.791 |
| persistence-only | 27.38 | -57.40 | 0.091 | +0.099 | 0.669 | 0.768 |
| full-control | 28.83 | -55.21 | 0.067 | +0.084 | 0.677 | 0.761 |

Paired: exploration-only vs no-control +0.103 (5/5), vs persistence-only +0.037 (4/5); full-control vs persistence-only −0.015 (0/5). **First genuinely promising exploration-specific result** — exploration-only improves block-switch adaptation, animal-like (early perseverative, later toward new prior). But full-control does NOT beat persistence-only, and the stale-switch probe remained negative. Why does exploration help alone but not when both controllers are on?

### Persistence/exploration interaction sweep (`..._interaction_sweep_v1/`)

Varying persistence/exploration scales inside full-control (5 seeds 42/123/456/789/2026, same block-switch budget):

| Condition | Persist | Explore | Retry gap | Block-switch lift | Δblock vs persist-only | +seeds | Δretry vs persist-only |
|-----------|---------|---------|-----------|-------------------|------------------------|--------|------------------------|
| persistence-only | 1.6 | off | 0.091 | +0.099 | baseline | - | baseline |
| exploration-only | off | 0.8 | 0.031 | +0.136 | +0.037 | 4/5 | -0.061 |
| full-control default | 1.6 | 0.8 | 0.067 | +0.084 | -0.015 | 0/5 | -0.025 |
| full-control persist-half | 0.8 | 0.8 | 0.067 | +0.136 | +0.037 | 5/5 | -0.025 |
| full-control persist-quarter | 0.4 | 0.8 | 0.097 | +0.096 | -0.003 | 2/5 | +0.006 |
| full-control explore-strong | 1.6 | 1.2 | 0.071 | +0.115 | +0.016 | 3/5 | -0.020 |
| full-control explore-double | 1.6 | 1.6 | 0.094 | +0.103 | +0.004 | 3/5 | +0.003 |
| full-control balanced | 0.8 | 1.2 | 0.082 | +0.112 | +0.013 | 3/5 | -0.009 |
| full-control explore-dominant | 0.4 | 1.6 | 0.103 | +0.112 | +0.013 | 4/5 | +0.011 |

`full_control_persist_half` matched exploration-only block-switch lift (+0.136), beating persistence-only by +0.037 in 5/5 — default full-control's persistence scale had *masked* the exploration effect. Caveat: it gives up retry-gap strength (−0.025, 0/5). **Honest status:** `persistence_only` remains the validated claim profile; `full_control_persist_half` is the best full-control exploration candidate for comparison/transfer; no single full-control setting yet dominates both. PRL/DMS should compare no-control, persistence-only, exploration-only, default full-control, and `full_control_persist_half`.

### What this achieved / did not

A legitimate controlled computational result: lesion-tested vs clean no-control; persistence-only recovers nearly the same retry mean as full control; psych/chrono stay calibrated; explicit "computational analogy not brain anatomy" caveat. **Not a breakthrough yet:** still inside the simulator; the retry metric was designed for this hypothesis (needs out-of-sample tasks); exploration not independently proven; the gate lesion softened but did not abolish the effect; no neural/anatomical claim. Status: a **strong internal milestone** — a reproducible, lesion-tested adaptive-control mechanism producing an animal-like weak-failure retry signature while preserving the evidence/RT profile.

---

## Reporting and Figures Pass (May 2026)

Documentation and figures, **not new science.** Commits `64e43d1`/`3096bdb`/`d51ac3a`/`273e25c`/`785bef8` (May 1–5).

- **README refocus** to lead with the validated claim (uncertainty-gated retry/persistence), with a "Scope of the current claim" block separating **supported** (weak-failure retry, 5/5, calibrated psych/chrono preserved) from **not yet validated** (rewarded-streak exploration — stale-switch went more negative under all conditions), a plain-language metrics key, and a Mermaid four-component diagram.
- **Journal-style figures** (`docs/figures/`): Fig 1 agent vs IBL mouse; Fig 2 suite summary (4-panel with IBL 20.0 ± 5.7 band, chrono ≈ −36 line); Fig 3 paired Δ vs no-control. Shared style helper `scripts/_figure_style.py`.
- **What this caught:** the chronometric panel originally used **mean** RT, inflated by rare IBL outliers (max 60,000 ms, mean 1,294 ms vs median 378 ms). Switching to **median RT with SE-of-median** (1.2533 × σ/√n) shrank the apparent agent–mouse RT gap and forced a README correction. Render scripts made self-bootstrapping.
- **Cleanup (`785bef8`):** removed dead one-off scripts, restored CI badge; 134 tests + `ruff` green at the May 5 push.

The validated claim is unchanged: uncertainty-gated retry/persistence supported across 5 seeds; rewarded-streak exploration not.

*(Registry milestone: 80+ experiments spanning Sticky-Q, PPO, Bayes Observer, Hybrid DDM+LSTM, R-DDM, and Adaptive-Control agents, Oct 2025 – May 2026.)*

---

## PRL Transfer and the Perseveration Mechanism (May 30, 2026)

### PRL transfer scaffold

Adaptive control rolls from its IBL-trained evidence core into Probabilistic Reversal Learning without changing the frozen `.ndjson` contract. **Key protocol correction:** PRL options stay visibly neutral — the richer option is a hidden contingency, payout probabilities reverse silently, and the policy receives no oracle reversal flag (it must infer change from reward outcomes). The env logs `reversal`, `block_index`, `contingency` for offline analysis only. The evaluator reports a PRL fingerprint: `optimal_choice_rate`, `reward_rate`, `early_optimal_choice_rate` (trials 1-5 post-reversal), `late_optimal_choice_rate` (6-10), `adaptation_lift = late − early`, `end_block_optimal_choice_rate` (final 20 pre-reversal), `block_learning_lift = end_block − early`, `stay_after_rewarded`, `switch_after_unrewarded`. A smoke run (`runs/prl_adaptive_control_smoke_codex/`, 170 trials, 2 reversals, optimal 0.424, adaptation_lift +0.100, epochs=0, one seed) is wiring evidence only, not a transfer claim. The DMS env is schema-valid and tested but intentionally one step behind (rollout/metrics unwired).

### Matched transfer result (5 conditions, 25 runs, 1,600 trials & 16 reversals/run, 40,000 trials total, 0/25 degenerate)

| Condition | Overall optimal | Reward rate | Early optimal | End-block optimal | Block-learning lift |
|-----------|-----------------|-------------|---------------|-------------------|---------------------|
| true no-control | 0.504 | 0.504 | 0.460 | 0.513 | +0.053 |
| exploration-only | 0.579 | 0.549 | 0.322 | 0.683 | +0.360 |
| persistence-only | 0.469 | 0.479 | 0.472 | 0.492 | +0.019 |
| full-control default | 0.507 | 0.501 | 0.510 | 0.466 | -0.044 |
| full-control persist-half | 0.510 | 0.502 | 0.478 | 0.543 | +0.066 |

The initial 10-trial window under-reported the phenomenon. `exploration_only` is deliberately perseverative right after a swap, then recovers over the block: end-block optimal +0.169 vs no-control and block-learning lift +0.307, both 5/5 seeds; vs persistence-only block-learning +0.341, 5/5. **First cross-task transfer, and the IBL dissociation REVERSES** — here exploration drives learning while persistence-only falls *below* no-control (perseveration). Arbitration result is negative and useful: no full-control profile preserves the exploration-only effect (`persist_half` improves overall optimal +0.042 in 5/5 but block-learning lift only +0.066 vs exploration's +0.360). **Claim boundary:** isolated exploration-specific transfer phenotype inside the simulator — NOT PRL animal parity (no PRL animal reference exists), and NOT a validated combined profile. Agents near chance absolutely.

### PRL arbitration scale sweep (10 conditions × 5 seeds, 50 runs, 80,000 trials, 0/50 degenerate)

| Full-control condition | Persist | Explore | Overall optimal | End-block optimal | Block-learning lift |
|------------------------|---------|---------|-----------------|-------------------|---------------------|
| default | 1.6 | 0.8 | 0.507 | 0.466 | -0.044 |
| persist-half | 0.8 | 0.8 | 0.510 | 0.543 | +0.066 |
| persist-quarter | 0.4 | 0.8 | 0.491 | 0.459 | -0.094 |
| explore-strong | 1.6 | 1.2 | 0.504 | 0.444 | -0.126 |
| explore-double | 1.6 | 1.6 | 0.511 | 0.534 | +0.057 |
| balanced | 0.8 | 1.2 | 0.503 | 0.445 | -0.080 |
| explore-dominant | 0.4 | 1.6 | 0.470 | 0.454 | -0.016 |

`exploration_only` stays at 0.579 / 0.683 / +0.360. Every full-control condition loses block-learning lift vs exploration-only in 5/5; saved checkpoints show scales stayed near their configured starts (not optimizer collapse). **Global scale tuning is not enough** — the combined controller does not learn the block even when persistence is reduced or exploration strengthened. Next: a sidecar reversal-window diagnostic (now implemented, `scripts/prl_arbitration_diagnostic.py`) logging control bias, persistence/exploration pressure, arbitration adjustment, and bounded residual to a separate `control_diagnostics.ndjson`.

### PRL perseveration mechanism confirmed

**Surface decomposition was misleading:** persistence/exploration pressures are ~0 in PRL for every condition, the exploration head is silent even in `exploration_only`, and `control_gate = 1.000` in every post-reversal window. PRL options are visually neutral, so `uncertainty = 1 − |contrast|` is pinned at 1.0 on every trial; the arbitration mixer barely acts (`arbitration_adjustment ~ 0.002`).

**The actual lever is `uncertain_retry` in `AdaptiveControlModel.update_plastic_history`** — "when uncertain and the last action failed, retry." In IBL, uncertainty varies with contrast so it only fires on ambiguous trials; in PRL, pinned at 1.0, it fires at full strength on every failure while the opposing `confident_switch` (scaled by `1 − uncertainty`) is dead → relentless perseveration.

**Controlled ablation (5 seeds, `uncertain_retry_enabled=False`, otherwise identical):**

| metric | `persistence_only` | `persistence_only` no-retry | `exploration_only` |
|---|---:|---:|---:|
| block_learning_lift | +0.019 | **+0.302** | +0.360 |
| optimal_choice_rate | 0.469 | **0.593** | 0.579 |
| end-of-block optimal | 0.492 | **0.657** | 0.682 |
| adaptation_lift | -0.035 | +0.123 | +0.070 |

All 5 seeds cleared the +0.20 threshold (paired deltas +0.069 to +0.441). **Two conclusions:** (1) the "exploration controller" never produced the `exploration_only` win — the head is inactive; `exploration_only` won *by subtraction* (it was the only lesion that disabled the misfiring `uncertain_retry`). (2) The root cause is a degenerate uncertainty signal, not arbitration — `uncertainty = 1 − |stimulus|` carries no information when the stimulus is always neutral. The principled fix (change-evidence, next section) supplies a volatility/reward-history uncertainty signal. Claim boundary unchanged: zero-shot agents still top out near 0.59 optimal choice; the confirmed result is mechanistic (perseveration via `uncertain_retry` is necessary and sufficient for the persistence deficit), not a PRL transfer success. `uncertain_retry_enabled` defaults to `True` (lesion knob; prior runs unaffected).

---

## Change-Evidence Recurrence and Cross-Task Calibration (May 31 – June 1, 2026)

### Recurrence implemented (May 31, flag-gated, default off)

The single uncertainty signal is split in two: `perceptual_uncertainty` (kept sensory `1 − |stimulus|`) and `change_evidence`, a decaying accumulator of recent committed failures (`λ·prev + (1−λ)·failure`, `change_evidence_decay` default 0.7). Opposite-direction gates: `retry_gate = perceptual·(1 − change_evidence)`, `switch_gate = clamp((1 − perceptual) + change_evidence)`. In IBL (low change_evidence in stable blocks) this reduces to current behavior; in PRL (perceptual pinned at 1.0) accumulated failures drive switching. The accumulator is essential — the 80/20 contingency means single-loss switching would fire on bad luck. Engineering: raw perceptual gate is model-owned and returned from `forward()`; recurrence threaded through plastic state with update-first timing, session/episode resets, TBPTT detach; base `HybridDDMModel` passes `change_evidence` through unchanged. **Flag-off is a verified bit-for-bit no-op** (IBL behavioral hash unchanged); flag-on confirmed live; sidecar logs `change_evidence`, `history_retry_gate`, `history_switch_gate`. Open concern flagged: at λ=0.7 the retry→switch crossover is the 2nd consecutive loss (~4% by chance on the good 80/20 option) — likely too eager; the recovery-sequence unit test passes; λ must be set by calibration. See `docs/prl_volatility_uncertainty_design.md`.

### Change-evidence calibration result (June 1) — pre-provenance-correction

*(Note: the retry-gap numbers in this subsection were computed before the June 1 metric provenance correction below and are **superseded for claim use**, retained for provenance.)* The mechanism is a credible cross-task rule; **λ=0.9 is the leading combined-profile candidate, not a new default** (feature opt-in, `change_evidence_enabled=False`).

Safety-gated λ calibration:

| λ | IBL persistence-only retry gap | IBL full-control retry gap | PRL run? | interpretation |
|---|---:|---:|---|---|
| flag off | 0.164 | 0.165 | historical | validated IBL reference |
| 0.70 | 0.052 | 0.066 | skipped | too eager; failed the IBL gate |
| 0.80 | 0.115 | 0.099 | yes | viable first rescue |
| 0.85 | 0.075 | 0.091 | skipped | dominated by 0.80 |
| 0.90 | 0.095 | 0.115 | yes | leading combined-profile candidate |

At λ=0.9 psych/chrono stayed healthy, 0 degenerate, 0 RT-ceiling; full-control retry improved over λ=0.8 (0.099→0.115) but remained below historical flag-off 0.165.

PRL transfer with `uncertain_retry` still enabled:

| condition | λ=0.80 optimal | λ=0.80 block-lift | λ=0.90 optimal | λ=0.90 block-lift |
|---|---:|---:|---:|---:|
| no control | 0.505 | +0.053 | 0.504 | +0.053 |
| exploration only | 0.700 | +0.249 | 0.704 | +0.232 |
| persistence only | 0.724 | +0.386 | 0.703 | +0.449 |
| full control default | 0.717 | +0.379 | 0.706 | **+0.469** |
| full control persist-half | 0.718 | +0.424 | **0.712** | +0.465 |

All λ=0.9 conditions beat no-control on block-learning lift in 5/5. The combined controllers **no longer suppress** PRL recovery — the recurrence fixes the mechanistic failure it was designed for (repeated failures close the retry gate and open the switch gate without a task-name special case).

**The PRL dissociation reverses under the recurrence (most important interpretive shift).** In the pre-recurrence May suite `exploration_only` was the sole driver (block-lift +0.360) while `persistence_only` fell below no-control (perseverated). Under λ=0.9 the ordering flips: `persistence_only` +0.449, full control +0.469, `exploration_only` trails at +0.232. The mechanism explains why — the earlier "exploration win" was winning *by subtraction* (it disabled the misfiring `uncertain_retry`); the recurrence un-breaks `uncertain_retry` directly, so persistence-based control recovers on its own. The recurrence's contribution is **repairing persistence under pinned uncertainty**, not demonstrating the combined agent requires exploration. Readers from the May transfer sections should treat this as **superseding the "exploration drives PRL" framing for the flag-on regime.** Across the flag-on sequence, 130 logs / 240,000 records passed schema validation (λ=0.7 IBL, λ=0.8 IBL+PRL, λ=0.85 IBL, λ=0.9 IBL+PRL). **Claim boundary:** promising evidence for one state-dependent rule across stable perceptual choice and hidden-contingency reversal; still in-simulator, not PRL animal parity; λ=0.9 opt-in until the IBL retry shortfall is understood or accepted.

---

## Adaptive Retry Metric Provenance Correction (June 1, 2026)

A focused λ=0.9 trace found an evaluator bug in `compute_adaptive_control_probe_metrics()`: the retry probe split weak vs strong failures using the **newly sampled current trial's** stimulus strength; it must use the **previous failed trial's** strength. Because IBL resamples contrast every trial, the old metric mixed unrelated trials. The evaluator now sorts within session, shifts absolute stimulus strength by one trial, and bins by that prior strength; a regression test pins the timing contract. The canonical IBL lesion suite was re-evaluated from saved `.ndjson`.

Corrected λ calibration (**these are the current claim-bearing numbers**):

| λ | IBL full-control retry gap | PRL full-control block-learning lift | interpretation |
|---|---:|---:|---|
| flag off | 0.175 | -0.044 | corrected historical baseline |
| 0.70 | 0.058 | not run | too eager |
| 0.80 | 0.111 | +0.379 | viable first rescue |
| 0.85 | 0.122 | not run | below λ=0.9 |
| 0.90 | **0.158** | **+0.469** | validated opt-in cross-task profile |

The corrected result is **stronger** than the pre-correction report: λ=0.9 nearly preserves historical full-control IBL retry (0.158 vs 0.175) while restoring combined PRL recovery with `uncertain_retry` still enabled, and exceeds the corrected historical `persistence_only` retry gap (0.120). **Feature remains default off**; PRL animal parity still unavailable; the corrected baseline must be used in future figures/prose; `persistence_only` remains the conservative standard IBL profile; **λ=0.9 is now the validated opt-in profile** for explicitly labeled cross-task experiments, not merely a candidate.

**Focused λ=0.9 trace finding:** a 10-reroll, 20,000-trial sidecar comparison confirmed the recurrence is live in IBL — mean `change_evidence` 0.239 after isolated losses and 0.334 after repeated losses; mean retry gate falls 0.680→0.596 while switch gate rises 0.337→0.428; after wins `change_evidence` decays gradually. Intended state-dependent behavior, not a dead flag.

**All pre-correction retry-gap values in older sections are retained only for provenance and are superseded for claim use.**

---

## DMS Memory Fingerprint Defined (June 1, 2026)

The DMS environment remains a schema-valid scaffold. The memory scorecard and lesion ladder are defined in `docs/dms_memory_fingerprint_design.md`: overall accuracy, delay-retention curve, stimulus-strength breakdown, match/non-match balance, omissions, reaction-time breakdown, a **memoryless baseline**, and a **delay-state reset lesion**. Adaptive rollout remains intentionally unwired until those controls exist.

---

## IBL Reference Expansion (July 2026)

`scripts/fetch_ibl_reference.py` pulls multi-session `biasedChoiceWorld` data from the IBL public server (OpenAlyx, anonymous) into the project schema, to test whether the reference fingerprint (currently 10 sessions / 8,406 trials) reproduces on an independent, larger sample. **Add-and-compare only**: `data/ibl/reference.ndjson` is untouched and no targets adopted.

### Pilot: 20 sessions (July 3) — baseline fingerprint replicates

20-session QC'd pull, 15,583 trials, seed 0:

| metric | baseline (10 sess) | expanded (20 sess) | within 1σ |
|---|---:|---:|:--:|
| psych slope | 19.97 ± 5.70 | 21.94 ± 13.42 (median 17.5) | ✓ |
| chrono slope (ms/unit) | -51.04 ± 63.65 | -27.67 ± 41.64 | ✓ |
| win-stay | 0.72 ± 0.08 | 0.69 ± 0.04 | ✓ |
| lose-shift | 0.47 ± 0.10 | 0.48 ± 0.08 | ✓ |
| lapse low / high | 0.08 / 0.10 | 0.07 / 0.06 | ✓ |

First independent cross-validation that the reference distribution is not an artifact of 10 hand-picked sessions. **Honest caveat:** more data did NOT tighten the psych-slope band (±5.70 → ±13.42). High-slope sessions (42–49) are genuinely proficient mice with near-zero lapse and near-vertical curves; with only five contrast levels the slope is weakly identified at the steep end. The 10-session baseline **under-sampled true between-mouse variation** (population median ~17, range 4–49). A wider band makes the hybrid agent's psych slope (~12–18) sit *more* comfortably within range. Report psych slope as **median + IQR**; win-stay tightened as expected (±0.08 → ±0.04).

**Two infrastructure bugs caught before adoption** (both "infrastructure correctness is a scientific concern"): (1) **Session selection** — OpenAlyx `biasedChoiceWorld` is ~73% trained / 27% low-performers and `one.search()` front-loads poor sessions, so `eids[:N]` grabbed near-chance mice (full-contrast accuracy ~0.68, near-flat psychometric). Fixed with a trained-performance QC gate (`--min-easy-accuracy`, default 0.85 on |contrast|=1.0) and a deterministic shuffle. (2) **Action integer convention** — reference logs and `eval.metrics.load_trials` use `0 = right, 1 = left`, the inverse of the env's `ACTION_LEFT=0 / ACTION_RIGHT=1`. Emitting env convention made `load_trials` relabel every right choice as left, inverting the psychometric (slope 0.15, bias 183, lapse 0.48) despite clean 0.9+ accuracy. History metrics (action-only) stayed healthy — which localized the fault to the choice-vs-contrast axis. Fixed by remapping at emission; an end-to-end regression test asserts a steep, low-lapse fit.

### 80-session scale + RT convention (July 4)

Scaled to 80 QC'd sessions (57,888 trials; all choice-sign agreement 1.000):

| metric | baseline (10) | expanded (80) |
|---|---:|---:|
| psych slope | 19.97 ± 5.70 (med 20.33) | 22.43 ± 12.99 (med 20.1, IQR 12.7–29.6) |
| win-stay | 0.72 (med 0.70) | 0.69 (med 0.70) |
| lose-shift | 0.47 (med 0.50) | 0.46 (med 0.46) |
| lapse low/high | 0.08 / 0.10 | 0.06 / 0.06 |

Psychometric, history, and lapse fingerprints reproduce at scale (psych median 20.1 vs 20.33). Report psych slope as median + IQR — the mean is inflated by proficient mice railing the fit's upper slope bound.

**Chronometric RT convention resolved — baseline uses `response_times`.** The chrono slope initially looked too shallow because the fetcher defaulted to `firstMovement_times − stimOn` (~155 ms median), whereas the baseline used `response_times − stimOn`. A matched 15-session A/B:

| source | median RT | RT-vs-\|contrast\| (0 → 1.0) | chrono slope |
|---|---:|---|---:|
| baseline | 379 ms | 540 → 309 | -15.6 |
| firstMovement | 156 ms | 220 → 112 | -8.6 |
| response | 405 ms | 539 → 320 | -16.8 |

`response` reproduces the baseline RT distribution and chrono slope almost exactly. The earlier "378 ms ⇒ firstMovement" inference was backwards — 378 ms is the response (movement-completion) measure. Fetcher default is now `--rt-source response`; `firstMovement` (the more standard decision-RT for chronometrics) remains available but does not match this project's calibrated targets. This is the RT-definition footgun flagged at the outset, now empirical.

**Status:** all six fingerprints reproduce on an 80-session independent sample under the correct RT convention. The reference distribution is not an artifact of the 10 hand-picked sessions. Still add-and-compare (`reference.ndjson` and its targets remain canonical); adopting a re-derived larger reference is the next deliberate, dated decision. Feature on `feat/ibl-reference-expansion` (PR #3), default-off in that nothing consumes the expanded pull yet.

---

*The full, unabridged chronological lab notebook — every original entry with all repetition preserved — is archived verbatim at `docs/archive/FINDINGS_raw_2026-07.md`.*
