# AnimalTaskSim Findings

**Benchmarking reinforcement learning agents against rodent and primate decision-making fingerprints**  
*October 2025 · Version 0.1.0 — Updated February 2026*

---

## Context

AnimalTaskSim compares learning agents to real animals on the IBL mouse 2AFC and the Roitman & Shadlen macaque random-dot motion (RDM) tasks. Every run logs one JSON object per trial under a frozen schema, enabling direct comparison of psychometric, chronometric, history, and lapse statistics. The October 2025 round of experiments focused on two priorities:

- Remove simulator shortcuts (auto-commit, implicit latency) that previously inflated agent resemblance.
- Regenerate baseline runs and document the resulting behavioral gaps using the hardened pipeline.

Fresh evidence comes from `runs/ibl_stickyq_latency/` (Sticky-Q with a 200 ms latency) and `runs/rdm_ppo_latest/` (PPO with collapsing bounds disabled). We also summarize the best-performing hybrid DDM+LSTM run (`runs/rdm_wfpt_regularized/`) to highlight what mechanism-level structure buys us.

---

## Methodological Note: RT Ceiling Saturation (February 2026)

> **Important caveat for interpreting chronometric slopes.**

Several runs report large negative chronometric slopes (e.g., −1813 ms/unit for `20251019_rdm_hybridddml`), but RT-by-coherence breakdowns reveal that low-coherence trials are pinned at the response window ceiling (typically 1200 ms). The slope is driven by a step function between ceiling-clamped slow trials and fast high-coherence trials, not by smooth evidence accumulation dynamics as in animal data.

The evaluation stack now includes two new metrics to flag this:

- **`ceiling_fraction`**: Fraction of difficulty levels where median RT equals the maximum observed RT. Values ≥ 0.5 indicate slope is unreliable.
- **`rt_range_ms`**: Range between fastest and slowest median RTs across levels.

| Run | Reported slope | `ceiling_fraction` | RT distribution |
| --- | --- | --- | --- |
| `20251019_rdm_hybridddml` | −1813 ms/unit | **0.50** (3/6 levels at 1200 ms) | Step function: 1200/1200/1200/880/540/370 |
| `hybrid_wfpt_curriculum` | −767 ms/unit | ~0.50 | Similar ceiling pattern |
| Macaque reference | −645 ms/unit | 0.0 | Graded: 302–525 ms, no ceiling |

**Takeaway:** When `ceiling_fraction ≥ 0.5`, the chronometric slope reflects response window limits, not evidence accumulation. True DDM-like chronometric behavior requires RTs that vary smoothly across all difficulty levels without hitting environment-imposed ceilings.

---

## Infrastructure Changes

- **Collapsing bound default**: `envs/rdm_macaque.py` now defaults to `collapsing_bound=False`, forcing agents to manage commitment timing.
- **Mouse latency hook**: `envs/ibl_2afc.py` exposes `min_response_latency_steps`; agents cannot act until the latency expires, but the environment no longer commits on their behalf.
- **Config persistence**: `agents/sticky_q.py` and `agents/ppo_baseline.py` serialize latency and reward-tuning fields into each run’s `config.json` for reproducibility.
- **JSON hygiene**: `scripts/evaluate_agent.py` and `eval/metrics.py` coerce NaN fits to `null`, keeping generated `metrics.json` files schema-compliant and flagging failed regressions explicitly.

These corrections ensure that reaction-time metrics reflect agent policy choices, not environment defaults.

---

## Mouse IBL 2AFC — Sticky-Q with Latency

- **Run**: `runs/ibl_stickyq_latency/`
- **Configuration highlights**: `min_response_latency_steps=20` (≈200 ms), 8 000 PPO-style gradient steps, deterministic seed 0.

| Metric | Agent | Reference (IBL) | Gap |
| --- | --- | --- | --- |
| Bias | −0.0001 | +0.074 | Matches magnitude but opposite sign (−0.074) |
| Psychometric slope | 33.3 | 13.2 | 2.5× steeper (overconfident choices) |
| Median RT (all contrasts) | 210 ms | 300 ms | 90 ms faster despite added latency |
| RT slope (ms/unit contrast) | −0.2 | −36.4 | Essentially flat (agent commits as soon as allowed) |
| Win-stay | 0.67 | 0.73 | Under-expresses win streaks |
| Lose-shift | 0.48 | 0.34 | Overreacts to errors |

### Interpretation (Mouse)

- The explicit latency lifts RTs into the right ballpark but does not create a coherence-dependent chronometric curve; Sticky-Q executes immediately when the gate opens.
- Choice slope remains too steep, indicating over-reliance on stimulus contrast without matching the animals’ lapse and bias mixture.
- Sequential dependencies still diverge: the agent over-perseverates on losses and under-perseverates on wins, suggesting its simple sticky prior cannot capture the asymmetric IBL history kernel.

Artifacts: `trials.ndjson`, `metrics.json`, `report.html`, and `dashboard.html` all live under `runs/ibl_stickyq_latency/`.

---

## Macaque RDM — PPO Baseline Re-evaluation

- **Run**: `runs/rdm_ppo_latest/`
- **Configuration highlights**: `collapsing_bound=False`, 200 000 timesteps, reward structure identical to prior releases (per-step cost disabled, accuracy reward only).

| Metric | Agent | Reference (Roitman & Shadlen) | Gap |
| --- | --- | --- | --- |
| Bias | +0.52 | ≈0 | Large pathological bias persists |
| Psychometric slope | 50.0 | 17.6 | 2.8× steeper |
| Median RT | 60 ms | 760 ms | 700 ms too fast |
| RT slope | 0 ms/unit | −645 ms/unit | No evidence-based slowing |
| Lapse rate (low coherence) | 0.49 | ~0 | Agent punts half the time to relieve decision pressure |

### Interpretation (PPO)

- Removing the collapsing bound exposes that PPO never learned to delay its response; it fires immediately and relies on random lapses to balance reward, yielding a flat chronometric curve.
- The positive bias indicates asymmetric value estimates that are not present in macaque behavior. Hyper-parameters that previously looked acceptable were benefiting from environment auto-commit; without it, the shortcomings are obvious.
- Reward shaping alone is insufficient; structural inductive bias is required to produce realistic RT distributions.

Artifacts: `runs/rdm_ppo_latest/trials.ndjson`, `metrics.json`, `report.html`, `dashboard.html`.

---

## Macaque RDM — Hybrid DDM+LSTM with Curriculum Learning

- **Run**: `runs/hybrid_wfpt_curriculum/`
- **Objective**: Demonstrate that a curriculum learning strategy focused on the WFPT likelihood can produce a negative chronometric slope.

| Metric | Hybrid Agent | Reference | Gap |
| --- | --- | --- | --- |
| RT intercept | 1.26 s | 0.76 s | 500 ms slower |
| RT slope | −767 ms/unit | −645 ms/unit | 19% overshoot |
| Psychometric slope | 7.33 | 17.6 | Too shallow |
| Bias | +0.001 | ≈0 | Matches |

### Interpretation (Hybrid)

- The curriculum learning strategy was successful. By prioritizing the WFPT loss in the initial phase of training, the agent was able to learn the fundamental relationship between evidence and reaction time, resulting in a strong negative chronometric slope.
- While the slope is a near-perfect match, the agent's reaction times are still globally slower than the macaques', and the psychometric slope is shallower. This suggests that further calibration of the non-decision time and drift-gain parameters is needed.
- This result provides strong evidence that a hybrid architecture with a mechanistic core, trained with a principled, curriculum-based approach, is a viable path toward replicating animal behavior.

Artifacts: `runs/hybrid_wfpt_curriculum/metrics.json`, `dashboard.html`.

---

## Macaque RDM — Hybrid DDM+LSTM with Time-Cost Guardrails

- **Run**: `runs/hybrid_wfpt_curriculum_timecost/`
- **Objective**: Preserve the negative chronometric slope while widening the agent’s response window and keeping WFPT loss dominant.

| Metric | Guardrailed Agent | Reference | Gap |
| --- | --- | --- | --- |
| RT intercept | 0.883 s | 0.76 s | +123 ms |
| RT slope | −267 ms/unit | −645 ms/unit | Slope regained but still shallow |
| Psychometric slope | 7.50 | 17.56 | Agent remains conservative |
| Bias | +0.24 | ≈0 | Small positive offset |
| History (win/lose/sticky) | 0.22 / 0.47 / 0.42 | 0.46 / 0.52 / 0.46 | Under-uses reward history |

### Interpretation (Time-Cost Curriculum)

- Extending the WFPT warm-up (15 epochs, heavier drift and non-decision supervision) plus a wider commit window (`max_commit_steps = 180`) prevents the chronometric curve from collapsing to the 1.2 s ceiling. The slope is again negative, though about 40% of the macaque magnitude.
- Reaction times are shorter overall (mean ≈810 ms), yet low coherence trials remain slower than the animals’, signalling that non-decision tuning or evidence noise still needs work.
- History metrics dipped below reference values. While the agent no longer locks onto a single action, it now under-perseverates; we will introduce explicit history losses or supervised pretraining to rebalance win-stay/lose-shift.

Artifacts: `runs/hybrid_wfpt_curriculum_timecost/trials.ndjson`, `metrics.json`, `training_metrics.json`, `curriculum_phases.json`, and `dashboard.html`.

---

## Macaque RDM — Soft RT Penalty Sweep

- **Runs**: `runs/hybrid_wfpt_curriculum_timecost_attempt1/`, `runs/hybrid_wfpt_curriculum_attempt2_two_phase/`, `runs/hybrid_wfpt_curriculum_timecost_soft_rt/`
- **Objective**: Stabilise the WFPT warm-up while nudging reaction times toward macaque means using a soft penalty instead of hard MSE.

| Configuration | Chronometric slope | RT intercept | Notes |
| --- | --- | --- | --- |
| Time-cost guardrail (Attempt 1) | ≈0 ms/unit | 1.20 s | RT ceiling hit; agent collapsed to a single side (bias ≈+6). |
| WFPT two-phase baseline | −505 ms/unit | 1.24 s | No RT penalty; slope healthy but intercept still high. |
| Soft RT (latest) | −165 ms/unit | 0.93 s | Soft penalty prevents ceiling camping, but slope remains shallow and history kernels under-shoot macaque values. |

### Interpretation (Soft RT Sweep)

- The soft penalty successfully avoids the 1.2 s clamp that flattened earlier runs, while keeping WFPT dominant in phase 1.
- Later phases still struggle to preserve a strong chronometric gradient: higher `rt_soft` weights drag the slope toward zero, whereas lower weights leave intercepts high.
- History metrics drop below macaque values when RT pressure increases; the agent becomes less perseverative and loses win-stay behaviour.

Artifacts and dashboards:

- `runs/hybrid_wfpt_curriculum_timecost_attempt1/dashboard.html`
- `runs/hybrid_wfpt_curriculum_attempt2_two_phase/dashboard.html`
- `runs/hybrid_wfpt_curriculum_timecost_soft_rt/dashboard.html`

Archived sweeps (`runs/archive/hybrid_wfpt_curriculum_timecost_soft_rt_attempt*/`) capture additional RT-weight combinations for forensic analysis.

---

## Cross-task Observations

- **Reaction-time realism requires policy-side latency.** Forcing a delay in the environment merely shifts intercepts. Agents need internal state or objectives that reward waiting for evidence.
- **Bias calibration is fragile.** Sticky-Q hits near-zero bias while the animals favor one choice slightly; PPO drifts heavily positive. Incorporating bias priors from the dataset or penalizing large offsets could help.
- **History kernels remain underfit.** Neither baseline reproduces the nuanced win-stay/lose-shift asymmetry or the decaying history kernels documented by the labs. Additional recurrent structure or explicit history penalties are required.
- **Schema validation protects analysis.** Forcing NaNs to `null` in metrics surfaces unstable regressions instead of hiding them, helping diagnose where modeling assumptions break (e.g., PPO’s chronometric slope fit fails because all RTs are identical).

---

## Outstanding Risks

- **Latency parameter tuning**: The current Sticky-Q run hard-codes 20 steps of latency. Different durations change RT intercepts substantially; we lack a principled calibration procedure.
- **Reward hacking in PPO**: With environment shortcuts gone, PPO absorbs penalty by random lapses. Without additional constraints, future hyper-parameter sweeps could land on superficially improved accuracy that remains behaviorally implausible.
- **Hybrid agent sustainability**: The WFPT loss and regularization pipeline is delicate. Training remains slow on CPU, and scaling to PRL/DMS will require careful batching.
- **Documentation drift**: Past READMEs overstated parity with animal data. This write-up replaces those claims, but future contributors must continue to run schema checks and update findings when new evidence arrives.

---

## RDM PPO Calibration Chronicle (Sept–Oct 2025)

| Iteration | Key Configs | Intent | Outcome |
| --- | --- | --- | --- |
| `rdm_ppo_avgcost_v1` | Avg-reward controller (`scale=0.5`) | Penalize long trials to induce RT slope | Agent froze on HOLD; chronometric curve remained flat and accuracy collapsed to 48 %. |
| `rdm_ppo_avgcost_v2` | Avg-reward controller (`scale=0.05`) | Softer penalty to restore action while nudging RTs | Accuracy recovered (≈60 %), but RTs still pegged at 60 ms; slope ≈0. |
| `rdm_ppo_calib_s005_*` | Avg-reward `0.05`, urgency feature {0.8, 1.2, 1.6} | Add policy-side urgency signal | RTs unchanged; high urgency destabilized policy (bias blow-up). |
| `rdm_ppo_calib_s002_*` | Avg-reward `0.02`, urgency {0.8, 1.2} | Reduce penalty, same urgency | Chronometric slope still zero; psychometric slope oscillated wildly. |
| `rdm_ppo_confidence_v1` | Confidence reward + RT target, no penalties | Reward waiting proportionally to evidence | RT intercept moved to 90 ms but slope stayed zero; policy learned to commit once hold expired. |
| `rdm_ppo_threshold_v1` | Evidence threshold gate + confidence reward | Force bound crossing before response | Accuracy moderate, yet RT stayed 60 ms because high coherence hits threshold instantly. |
| `rdm_ppo_threshold_v2` | Slower noise / higher gain adjustments | Make evidence accumulation harder | Psychometric slope overshot to 6× reference; RT still flat. |
| `rdm_ppo_hold{20,30}_*` | Response hold 20–30 steps + thresholds 1.5–3.0 | Simulate motor prep window | Intercepts rose (210–310 ms) but coherence effect absent; high thresholds tanked reward. |
| `rdm_ppo_thresh3_v3` | Threshold 3.0, no hold, low evidence gain | Probe extreme gating | Policy reverted to immediate commits; accuracy ≈ chance. |

### Lessons Learned

- **Environment gating without stimulus dynamics fails.** Coherence levels need to modulate evidence flow over many steps; otherwise, agents cross the bound almost immediately.
- **Reward shaping alone cannot buy chronometry.** Bonuses tied to cumulative evidence or RT targets simply shift intercepts while keeping zero slope.
- **Urgency signals need conflict.** With no cost to instant action, the agent ignores urgency features or treats them as noise.

### October 11 Sweep: Stimulus Pacing + Time-Cost Probes

After instrumenting `envs/rdm_macaque.py` with coherence-dependent sampling, optional trace logging, and Tyro-exposed overrides for stimulus/response durations, we ran three focused PPO experiments to bend the chronometric curve without artificial holds.

| Run | Key Config (diffs relative to baseline) | Psychometric | Chronometric (ms by coherence) | History snapshot |
| --- | --- | --- | --- | --- |
| `rdm_ppo_coherence_long_20251011` | Stimulus 160, response 200, `use_coherence_dependent_sampling=True`, hold=10 | slope 3.35, bias ≈0, lapses ≈0 | flat 200 / 200 / 200 / 200 / 200 / 200 | win-stay 0.51, lose-shift 0.47, sticky 0.52 |
| `rdm_ppo_coherence_hold40_20251011` | Response 280, hold=40, lower gain floor | slope 50.0, bias −0.50, lapse_high 0.45 | flat 400 / … / 400 | win-stay 0.44, lose-shift 0.67, sticky 0.37 |
| `rdm_ppo_coherence_soft_20251011` | Hold=5, threshold 4.0, negative per-step reward | slope 6.0, bias −0.37, lapse_high 0.21 | flat 250 / … / 250 | win-stay 0.65, lose-shift 0.53, sticky 0.55 |
| `rdm_ppo_coherence_cost_20251011` | Negative per-step cost −5e-4, confidence bonus 0.5 | slope 1.63, bias +0.17 | flat 100 / … / 100 | win-stay 0.44, lose-shift 0.53, sticky 0.45 |
| `rdm_ppo_coherence_cost2_20251011` | Avg-reward scale 0.14, `bound_threshold=3.5`, no confidence | slope 6.77, bias −0.44, lapse_high 0.16 | flat 150 / … / 150 | win-stay 0.76, lose-shift 0.41 |

None of the variants produced the desired RT gradient; introducing negative per-step incentives amplified waiting without linking latency to coherence, while large holds simply pinned RT at the enforced duration and distorted choice history. These runs confirm that pacing must emerge from evidence accumulation (gain/sigma schedules) plus calibrated time costs, not from hard response blocks or reward bribes.

To isolate time-cost mechanisms we executed a three-run micro-sweep with 300 k PPO steps and 600-trial episodes:

| Run | Time pressure | Psychometric | Chronometric `rt_by_level` | Notes |
| --- | --- | --- | --- | --- |
| `rdm_ppo_chrono_A` | Avg reward (`scale=0.10`), `bound_threshold=3.0`, `min_bound_steps=2`, urgency 0.8 | slope 18.1, bias −5.85, lapse_low 0.16 | {0.0:20, 0.032:20, 0.064:30, 0.128:30, 0.256:25, 0.512:30} | Strong WSLS lock-in (win-stay=1.0) shows policy saturates a single action despite slight RT spread. |
| `rdm_ppo_chrono_B` | Avg reward (`scale=0.14`), tighter bound 2.5 | slope 3.69, bias ≈0, lapses ≈0 | All coherences 10 ms | Time penalty too strong—agent fires instantly to avoid cost. |
| `rdm_ppo_chrono_C` | Explicit time-cost controller (`base=3e-4`, `growth=0.01`) | slope 3.31, bias −0.03 | {0.0:30, 0.032:30, 0.064:30, 0.128:30, 0.256:20, 0.512:30} | Slight downward dip at 0.256, but still far from macaque slope. |

Takeaways:

- Avg-reward penalties require careful tuning; too low and policy camps on one choice, too high and it commits immediately.
- Time-cost controller yielded the only non-flat bin (20 ms at 0.256), hinting that collapsing bounds + cost can create a slope once evidence dynamics are further differentiated.
- History metrics highlight when policies exploit the reward structure instead of integrating evidence—monitor `win_stay`, `lose_shift`, and regression betas after every sweep.

### Next Plan of Record (Nov 2025)

1. **Stimulus pacing overhaul**: extend stimulus phase and stream motion pulses whose variance scales with coherence so low-coherence trials require longer integration.
2. **Average-reward per second objective**: switch PPO reward to `correct_reward - lambda * RT` with lambda calibrated to macaque mean reward rate; learnable non-decision time parameter to encourage evidence-based timing.
3. **History-informed baselines**: incorporate recent trial outcomes into policy inputs (or an auxiliary loss) to match win-stay/lose-shift kernels while maintaining schema stability.
4. **Benchmark harness**: document each calibration sweep in `runs/calibration_logs/README.md` with configs and evaluation plots to track progress and prevent repetition.

These steps refocus effort on stimulus realism and policy incentives rather than brute-force hyperparameter sweeps.

---

---

## Synthesis: The Three Agent Archetypes (Registry Analysis)

Analysis of our 21 registry entries reveals three distinct architectural families, each capturing different aspects of animal behavior while failing on complementary dimensions. This pattern suggests a clear path forward.

### Archetype A: Reward Optimizers (PPO Baseline)

**Representative runs**: `rdm_ppo_latest`, `20251017_ibl_ppo`, `20251018_ibl_ppo`

**Signature behavior**:

- Psychometric slopes: 50.0 (hyper-steep, near-deterministic)
- Chronometric slopes: ≈0 ms/unit (perfectly flat RT curves)
- Bias: highly variable (−207 to +0.52), unstable across tasks
- History effects: moderate win-stay (0.55–0.78) as reward exploitation artifact

**What they capture**: Task reward structure and asymptotic accuracy ceiling.

**What they miss**: Evidence accumulation dynamics, speed-accuracy tradeoffs, coherence-dependent timing, stable choice biases.

**Failure mode**: When environment shortcuts are removed (collapsing bounds, auto-commit), PPO agents either fire instantly (RT=60 ms) or adopt pathological repetition strategies (bias=−207, win-stay=1.0). The `rdm_ppo_latest` run exemplifies this: lapse rate reaches 49% on low-coherence trials as the agent punts decisions to maintain average reward.

### Archetype B: History-Biased Heuristics (Sticky-Q)

**Representative runs**: `ibl_stickyq_latency`, `20251017_ibl_stickyq`, `20251018_ibl_stickyq`

**Signature behavior**:

- Psychometric slopes: 30.3–33.3 (steep but more realistic than PPO)
- Chronometric slopes: ≈0 ms/unit (flat, like PPO)
- Bias: near-zero (−0.0001 to +0.005), stable across seeds
- History effects: strong (win-stay 0.54–0.67, lose-shift 0.38–0.60, sticky 0.52–0.67)

**What they capture**: Inter-trial dependencies (win-stay, lose-shift, choice stickiness) matching qualitative animal patterns.

**What they miss**: Intra-trial dynamics. RT remains flat regardless of stimulus difficulty; the 200 ms latency in `ibl_stickyq_latency` merely shifts the intercept without creating a coherence gradient.

**Critical insight**: Sticky-Q demonstrates that explicit history terms can replicate sequential biases, but tabular Q-learning with hand-engineered stickiness cannot learn to wait for evidence. The architecture has no mechanism to trade speed for accuracy.

### Archetype C: Mechanistic Integrators (Hybrid DDM+LSTM)

**Representative runs**: `hybrid_wfpt_curriculum`, `rdm_wfpt_regularized`, `20251017_rdm_hybridddml`, plus 13 curriculum/calibration variants

**Signature behavior**:

- Psychometric slopes: 5.1–32.3 (variable, often too shallow for macaque, too steep for mice with proper priors)
- Chronometric slopes: −165 to −1828 ms/unit (consistently negative, capturing speed-accuracy tradeoff)
- Bias: near-zero when stable (−0.002 to +0.43), but prone to collapse (+6.06 in pathological runs)
- History effects: **consistently near chance** (win-stay 0.12–0.52, lose-shift 0.28–0.52, sticky 0.48–0.62)

**What they capture**: The core decision process. These agents slow down for hard trials, speed up for easy ones, and can produce chronometric slopes that match (or overshoot) animal magnitudes. The best run (`hybrid_wfpt_curriculum`) achieves RT slope = −767 ms/unit vs. macaque reference −645 ms/unit.

**What they miss**: Inter-trial memory. Across all 13+ hybrid runs, history metrics hover around 0.5, indicating the LSTM "coach" is not learning to bias the DDM based on previous outcomes. The recurrent module sets initial DDM parameters but does not carry forward reward/choice history effectively.

**Failure modes**:

1. **RT ceiling collapse** (`hybrid_wfpt_curriculum_timecost_attempt1`): Agent camps at max allowed RT, chronometric slope flattens, bias explodes to +6.06, sticky rate hits 0.998 (pathological repetition).
2. **History washout** (`hybrid_wfpt_curriculum_timecost_soft_rt`): Introducing RT penalties to lower intercepts also erases history effects; win-stay drops to 0.16, lose-shift to 0.35.
3. **Shallow psychometrics** (most runs): Curriculum learning successfully produces negative RT slopes but leaves choice curves too flat (slopes 5–8 vs. reference 17.6), suggesting drift-gain calibration or lapse terms need tuning.

---

## The Decoupling Problem

**Core finding**: We have successfully modeled the two key behavioral phenomena in isolation but not in a single agent.

- **Intra-trial dynamics** (how a decision unfolds in time): Solved by Hybrid DDM+LSTM via curriculum learning on WFPT loss. Chronometric slopes are reliably negative.
- **Inter-trial dynamics** (how one trial influences the next): Solved by Sticky-Q via explicit history terms. Win-stay/lose-shift patterns match qualitative trends.

**Why hybrid agents fail on history**: The current architecture (`hybrid_ddm_lstm`) uses the LSTM to set static DDM parameters (drift rate, bound, non-decision time) at trial start. This is insufficient for creating history-dependent choice biases because:

1. The LSTM receives a compressed trial history but has no direct pathway to bias the evidence accumulation process dynamically.
2. The DDM parameters are frozen once set; subsequent evidence integration proceeds independently of past rewards/choices.
3. Training focuses on WFPT likelihood (which rewards accurate RT distributions) and task reward (which rewards correct choices), neither of which explicitly penalize failure to carry forward trial history.

---

## Path Forward: Architectural Recommendations

### Primary Recommendation: Recurrent Drift-Diffusion Model (R-DDM)

**Motivation**: Based on registry evidence, the LSTM must influence drift *during* the decision, not just at initialization.

**Proposed architecture**:

- At each time-step *t*, the RNN hidden state *h_t* (which encodes trial history) combines with current sensory evidence *e_t* to compute the instantaneous drift rate: `drift_t = f(h_t, e_t, coherence)`.
- This allows history to exert a *dynamic bias*: e.g., after a rewarded left choice, initial drift favors left, but strong rightward evidence can overcome this bias as the trial progresses.
- The RNN learns to modulate drift based on recent outcomes, naturally producing win-stay/lose-shift effects without hand-coded sticky terms.

**Expected outcome**: An R-DDM has the representational capacity to simultaneously:

1. Produce negative chronometric slopes (by modulating drift based on evidence strength, already proven in static hybrid runs).
2. Produce realistic history biases (by modulating drift based on recurrent memory state, the missing ingredient).

**Implementation path**:

1. Start with the proven WFPT curriculum from `hybrid_wfpt_curriculum`.
2. Replace the static DDM parameter predictor with a recurrent drift module: `drift_t = LSTM(h_t) + gain * evidence_t`.
3. Add auxiliary loss to penalize deviations from target win-stay/lose-shift rates during training.
4. Monitor both chronometric *and* history metrics in each epoch to ensure no regression.

### Incremental Alternative: History-Aware Hybrid Agent

If a full R-DDM is too large a step, augment the existing `hybrid_ddm_lstm` with explicit history inputs.

**Rationale**: The LSTM may not be learning the relevance of trial history from raw observation/reward sequences. Make history "first-class" by providing:

- `prev_choice` (one-hot: left/right)
- `prev_reward` (binary: 0/1)
- `prev_stimulus_strength` (float: coherence or contrast)

**Implementation**:

- Concatenate this 3-vector to the LSTM input at trial start.
- Optionally add an auxiliary loss: `L_history = MSE(predicted_win_stay_rate, target_win_stay_rate) + MSE(predicted_lose_shift_rate, target)`.
- Retrain existing checkpoints with this augmented input and monitor whether history metrics rise from 0.5 baseline.

**Risk**: This is a band-aid. If the fundamental issue is that DDM parameters are set statically, adding more input features may not solve the problem. However, it's a low-cost experiment that could yield useful diagnostics.

---

## Updated Findings (October 2025)

### R-DDM infrastructure & results

- Added end-to-end support for both IBL and macaque tasks: dataset loaders, task-aware rollouts, CLI wizard integration, and best-checkpoint evaluation (`--use-best`).
- `scripts/rddm_sweep.py` now explores drift supervision, history weights, and prior scaling. Sweeps recorded in `runs/rddm_sweep_ibl/` and `runs/rddm_sweep_rdm/`.
- Despite the tooling, R-DDM still fails to reproduce animal fingerprints. Best IBL runs reach psychometric slope ≈6 with flat chronometric slope and saturated history metrics; macaque runs plateau around slope ≈0.1. Future work must focus on stronger stimulus supervision / entropy regularisation.

### Hybrid DDM+LSTM status

- Historic run `20251019_rdm_hybridddml` (default 7-phase curriculum) remains the benchmark: psychometric slope ~32, chronometric slope ~–1812 ms/unit.
- A simplified three-phase sweep (`scripts/hybrid_sweep.py`, `runs/hybrid_sweep_rdm/`) showed that removing phases destroys the fingerprint (slopes 2–4, flat RT). Message: keep the full multi-phase curriculum or modify it in-place.
- `scripts/train_hybrid_curriculum.py` gained optional `--phase1-max-slope` / `--phase2-max-slope` arguments so custom curricula can enforce slope ceilings without rewriting the training loop.

### Registry snapshot (55 entries)

- **IBL Mouse 2AFC**: Sticky-Q, PPO baselines, extensive hybrid curriculum variants, and the new R-DDM experiments (curriculum + sweep).
- **Macaque RDM**: PPO/Bayes baselines, historical hybrid run, new hybrid sweep results, and R-DDM curriculum + sweep runs.
- Sweep-specific registries live in `runs/rddm_sweep_ibl/registry.json` and `runs/hybrid_sweep_rdm/registry.json` for easier filtering.

### Reference & evaluation assets

- Reference data unchanged (`data/ibl/reference.ndjson`, `data/macaque/reference.ndjson`).
- Evaluation stack includes `scripts/evaluate_agent.py`, `scripts/make_dashboard.py`, and the two sweep utilities.

---

## Next Steps (Prioritised)

1. **Stimulus-sensitive R-DDM**: Introduce direct drift supervision / entropy bonuses so the agent cannot ignore contrast while history/prior losses remain active.
2. **Curriculum-aware sweeps**: Modify `CurriculumConfig.history_finetune_curriculum()` to expose tunable weights (rather than replacing it) so sweeps can safely explore history/RT tuning without losing chronometric behaviour.
3. **Bias & lapse regularisers**: Hybrid runs still show unstable bias; add soft priors or penalties in late curriculum phases.
4. **Automation hygiene**: When running sweeps, always call `scan_runs.py --runs-dir … --registry-path …` so results remain indexed.
5. **Roadmap alignment**: Ensure the new R-DDM and sweep infrastructure can extend to PRL / DMS tasks while preserving schema contracts.

---

## Per-Trial History Loss — Decoupling Fix (February 2026)

### Root Cause Analysis

Deep inspection of both trainers revealed why models learn chronometric slopes but fail on history metrics:

| Trainer | History mechanism | Gradient quality |
|---------|------------------|------------------|
| **R-DDM** | `_history_regulariser`: batch-mean MSE `(E[stay\|win] - target)²` | Differentiable but weak — each trial gets O(1/N) gradient signal |
| **Hybrid** | `_estimate_history`: hard argmax on detached `prob_buffer` (list of floats) → NumPy tallies | **Zero gradient** — computation graph severed by `.detach()` + argmax |

The batch-mean approach (R-DDM) only constrains the mean of the stay/shift distribution. A model can satisfy it with any variance pattern. The Hybrid approach provides no learning signal at all.

### Fix: `per_trial_history_loss()`

Added to `agents/losses.py`. Key properties:

1. **Per-trial MSE** instead of batch-mean MSE: `mean((stay_prob_i - target)²)` vs `(mean(stay_prob_i) - target)²`. By Jensen's inequality, per-trial MSE ≥ batch-mean MSE — it penalises both mean deviation AND variance.
2. **Differentiable** through `choice_prob` to the model parameters.
3. **Convention-aware**: Supports both R-DDM (`no_action_value=-1`) and Hybrid (`no_action_value=0`) prev_action encodings.

### Changes

- `agents/losses.py`: `per_trial_history_loss()` function + `per_trial_history` weight in `LossWeights`
- `agents/r_ddm/config.py`: `per_trial_history_weight = 0.5` (ramped with same schedule as existing history loss)
- `agents/r_ddm/trainer.py`: Integrated as additive loss term in `_compute_losses`
- `agents/hybrid_ddm_lstm.py`: Added `prob_tensor_buffer` (keeps computation graph alive) alongside existing detached `prob_buffer`; integrated per-trial loss with Hybrid encoding convention
- `tests/test_per_trial_history.py`: 12 tests covering zero-loss, gradient flow, both conventions, edge cases, Jensen's inequality

### Remaining validation

Retrain R-DDM and Hybrid with `per_trial_history_weight > 0` and verify that win-stay/lose-shift metrics rise from ~0.5 without regressing chronometric slope. This is the key experiment.
6. ~~**WFPT normalization**: The WFPT density over-integrates for strong drift + wide bound~~ — **FIXED** (Feb 2026). Image charge positions in small-time series corrected from `z + 2ka` to `a(z + 2k)`. Both series now agree to 6 decimal places.
7. **RT ceiling mitigation**: Address the ceiling saturation pattern where low-coherence RTs are pinned at the response window maximum. Options: (a) increase max response window, (b) penalize ceiling-saturated RTs during training, (c) add ceiling-aware chronometric fitting that excludes clamped levels.

---

## R-DDM Formal Evaluation (February 2026)

Formal evaluation was run on `runs/r_ddm_choice_only_v4` — previously the most promising R-DDM training run based on training-time metrics (77.7% accuracy, win-stay 0.77).

### Best checkpoint (`trials_best.ndjson`)

| Metric | R-DDM (best) | IBL Reference | Notes |
| --- | --- | --- | --- |
| Psychometric slope | 15.07 | 13.2 | Closest match of any agent |
| Bias | 3.72 | +0.074 | Moderate rightward bias |
| Win-stay | **0.953** | 0.73 | Extreme perseveration |
| Lose-shift | 0.071 | 0.34 | Nearly zero — ignores errors |
| Sticky choice | 0.939 | — | Pathological stickiness |
| RT (all levels) | 300.0 ms (flat) | 300 ms median | All RTs identical (motor delay only) |

### Regular rollout (`trials.ndjson`)

| Metric | R-DDM (regular) | IBL Reference |
| --- | --- | --- |
| Psychometric slope | 5.41 | 13.2 |
| Bias | 5.96 | +0.074 |
| Win-stay | 0.865 | 0.73 |
| Sticky choice | 0.919 | — |
| p(right) | 0.14 | ~0.5 |

### Interpretation

The R-DDM shows **extreme choice stickiness** (win-stay >0.95, sticky >0.93) far exceeding animal levels. While training-time metrics reported balanced win-stay (0.77), the rollout behavior reveals a policy that locks onto a single action (p_right=0.04 for best checkpoint). All RTs are pinned at 300 ms (the motor delay floor), confirming the model learned no evidence accumulation dynamics. The `is_choice_only` training mode (WFPT weight=0) means no RT signal was available.

Despite the choice-only config, the psychometric slope on the best checkpoint (15.07) is the closest to the IBL reference (13.2) of any agent tested. This suggests the R-DDM architecture can learn reasonable stimulus sensitivity, but the extreme perseveration and absent RT dynamics prevent it from being a valid behavioral model.

---

## WFPT Implementation Audit (February 2026)

Unit tests added in `tests/test_wfpt.py` (20 tests) validated the WFPT likelihood implementation against analytical DDM properties.

### Findings

1. **Drift convention is inverted**: Positive drift increases P(choice=0), opposite to the standard DDM convention where positive drift favours the upper boundary (choice=1). The training pipeline compensates by learning inverted drift signs, so end-to-end results are unaffected. Documented in test file.

2. **~~Density normalization degrades for strong parameters~~ (FIXED)**: The small-time series had incorrect image charge positions (`z + 2ka` instead of `a(z + 2k)`), causing the density to misscale by a factor of `a²` when bound ≠ noise. This produced over-integration (e.g., ∫density = 2.4 for drift=3, bound=2). Fixed in February 2026; both series now agree to 6 decimal places and integrate to 1.0 for all tested parameter regimes.

3. **Edge cases are handled**: Near-zero drift, extreme biases (0.02, 0.98), very small and very large RTs all produce finite log-likelihoods. Gradient flow is verified for all 5 DDM parameters.

4. **Symmetry is preserved**: Flipping drift sign correctly mirrors choice likelihoods for unbiased starting points, and zero drift produces equal likelihoods for both choices.

---

## The Decoupling Experiment (February 2026)

### Motivation

The central scientific question of AnimalTaskSim is **the Decoupling Problem**: no agent simultaneously captures intra-trial dynamics (chronometric slope — slower RTs for harder stimuli) AND inter-trial dynamics (history effects — win-stay, lose-shift). Agents that learn good RT structure show random history patterns; agents tuned for history show flat chronometric curves.

We hypothesised that batch-mean history supervision (MSE over all trials in a batch) allows the model to satisfy the loss by matching *average* history statistics while individual trials show uncorrelated behaviour (Jensen's inequality: `MSE(mean) ≤ mean(MSE)`). A **per-trial history loss** that penalises each trial's win-stay/lose-shift deviation independently should close this gap.

### Architecture: Per-Trial History Loss

```
Standard:   L_history = MSE(batch_mean_WS, target_WS)
Per-trial:  L_history = (1/N) Σᵢ MSE(WS_trial_i, target_WS)
```

Implemented in `agents/losses.py::per_trial_history_loss()`, with weight `per_trial_history` in `LossWeights`. Value is propagated through differentable `prob_tensor_buffer` → per-trial softmax choice probabilities → per-trial win-stay computation.

### Phase 1: R-DDM Experiments (A–D)

Four runs with the R-DDM agent varying the per-trial history loss weight:

| Experiment | History Weight | Per-Trial Weight | WS (rollout) | Chrono Slope | Notes |
| --- | --- | --- | --- | --- | --- |
| A: Control | 0.5 | 0.0 | 0.53 | 0.0 ms/unit | RT pinned at 300ms |
| B: Per-trial only | 0.0 | 0.5 | 0.53 | 0.0 ms/unit | Identical to A |
| C: High per-trial | 0.0 | 2.0 | 0.53 | 0.0 ms/unit | Identical to A |
| D: Combined | 0.5 | 0.5 | 0.53 | 0.0 ms/unit | Identical to A |

**Result: No effect.** All four runs produced identical metrics on rollout. Root cause: R-DDM trains on *animal data* (supervised). The model already achieves near-zero loss on training data (WS=0.73 during training), so the per-trial loss has nothing to fix. The generalization gap (train: WS=0.73, rollout: WS=0.53) is a distribution-shift problem that per-trial supervision cannot address because the loss is already minimised on the training distribution.

**Lesson**: Per-trial history loss can only help agents that train on their *own* policy rollouts, not on fixed animal data. This motivated the Hybrid experiment.

### Phase 2: Hybrid DDM+LSTM Experiments (E–F)

The Hybrid agent trains on its own rollouts inside the RDM environment, so the per-trial loss directly constrains generative behaviour. Full 7-phase curriculum with seed=42, 10 episodes, 30 sessions.

| Metric | E: Control (per_trial=0.0) | F: Treatment (per_trial=0.5) | Direction | Animal Target |
| --- | --- | --- | --- | --- |
| Psychometric slope | 6.76 | 7.16 | +6% | ~10-20 |
| Chrono slope (ms/unit) | −19.3 | −26.6 | **+38%** | −100 to −300 |
| RT range (ms) | 110 | 155 | **+41%** | 200-400 |
| Win-stay | 0.176 | 0.194 | +10% | 0.6-0.8 |
| prev_correct_beta | −0.546 | −0.124 | **+77%** | >0 |
| p(right) | 0.165 | 0.172 | — | ~0.5 |
| Lapse (high) | 90.3% | 90.4% | — | <15% |

### Interpretation

**Directional success, magnitude insufficient.**

The per-trial history loss produces consistent improvements on every key metric:

- Chronometric slope steepened by 38% (better intra-trial dynamics)
- RT range widened by 41% (less ceiling saturation)
- `prev_correct_beta` improved from −0.55 to −0.12 (agent less anti-correlated with correct direction)
- Win-stay improved 10% (but still far below animal levels)

However, both runs share a fundamental calibration problem: ~83% leftward choice bias and ~90% lapse on the right side. This is a curriculum-level issue, not a per-trial history issue — the 7-phase curriculum with only 10 episodes and 30 sessions doesn't produce well-calibrated psychometric baselines.

### Conclusions & Next Steps

1. **The mechanism works**: Per-trial history loss improves all Decoupling metrics in the predicted direction. The Jensen's inequality hypothesis is supported.

2. **Base model quality gates the result**: With n=10 episodes and 30 sessions, neither run achieves balanced psychometric performance. The per-trial signal is overshadowed by the massive bias.

3. **Recommended next experiment**: Run with higher training budget (episodes=50, max-sessions=100) and verify the base model achieves p(right) ≈ 0.5 before comparing history effects. Also test `per_trial_history_weight` = {0.1, 0.25, 0.5, 1.0} to find optimal strength.

4. **Broader implication**: The fact that R-DDM (supervised on animal data) showed zero effect while Hybrid (self-play) showed directional improvement validates the architectural distinction. An agent can only be *constrained* by a loss on behaviour it actually generates.

### Phase 3: Large-Budget Hybrid Experiments (G–H)

Scaled to 30 episodes, 80 sessions, 5 history-phase epochs. Same seed=42, same curriculum.

| Metric | G: Control (per_trial=0.0) | H: Treatment (per_trial=0.5) | Direction | Animal Target |
| --- | --- | --- | --- | --- |
| Psychometric slope | 6.83 | 6.71 | −2% | ~10-20 |
| Chrono slope (ms/unit) | −24.2 | −18.5 | **−24%** | −100 to −300 |
| RT range (ms) | 140 | 120 | **−14%** | 200-400 |
| Win-stay | 0.173 | 0.175 | +1% | 0.6-0.8 |
| Lose-shift | 0.370 | 0.381 | +3% | 0.3-0.5 |
| prev_correct_beta | +0.360 | +0.074 | **−79%** | >0 |
| p(right) | 0.164 | 0.165 | — | ~0.5 |

### Full Cross-Budget Comparison (E–H)

| Metric | E: Ctrl (10ep) | F: Treat (10ep) | G: Ctrl (30ep) | H: Treat (30ep) | Animal |
| --- | --- | --- | --- | --- | --- |
| p(right) | 0.165 | 0.172 | 0.164 | 0.165 | ~0.5 |
| Psych slope | 6.8 | 7.2 | 6.8 | 6.7 | 10-20 |
| Chrono slope | −19.3 | −26.6 | −24.2 | −18.5 | −100+ |
| RT range (ms) | 110 | 155 | 140 | 120 | 200+ |
| Win-stay | 0.176 | 0.194 | 0.173 | 0.175 | 0.6+ |
| prev_correct_beta | **−0.546** | −0.124 | **+0.360** | +0.074 | >0 |

### Updated Interpretation

**The small-budget directional effect did not replicate at larger scale.**

In Phase 2 (E/F, 10 episodes), the per-trial loss improved every metric. In Phase 3 (G/H, 30 episodes), the effect **reversed**: the control outperforms the treatment on chronometric slope (−24.2 vs −18.5), RT range (140 vs 120), and prev_correct_beta (+0.36 vs +0.07). This means:

1. **The E/F result was likely noise**, not signal. With ~2000 rollout trials per run, metric estimates have high variance. The apparent "directional improvement" was within the noise floor.

2. **More training naturally solves prev_correct_beta** — the control improved from −0.546 (E) to +0.360 (G) purely from 3x more training, with no per-trial loss. This is the biggest finding: the correlation between previous reward and next choice emerges from training duration, not from per-trial supervision.

3. **The per-trial loss may actually interfere** with other objectives at longer training horizons. The treatment's worse chrono slope and prev_correct_beta suggest the per-trial gradient competes with the WFPT/drift supervision signals.

4. **The leftward bias remains the dominant problem.** All four runs have p(right) ≈ 0.16 — an ~84% leftward bias that makes all other metrics unreliable. No amount of history supervision matters when the agent barely explores one side of the choice space. This is a curriculum/architecture issue that must be fixed before the Decoupling experiment can yield meaningful conclusions.

### Revised Conclusions

1. **Per-trial history loss is not the solution** to the Decoupling Problem, at least not in isolation. The E/F signal was not robust.

2. **Training budget helps more than loss engineering.** The natural improvement in prev_correct_beta from E→G (+0.9 units) dwarfs anything the per-trial loss contributed.

3. **The real bottleneck is choice bias.** Fixing p(right) ≈ 0.5 is prerequisite to any meaningful Decoupling experiment. Potential fixes:
   - Stronger choice loss weight during early phases
   - Explicit anti-bias regularisation (penalise p(right) deviation from 0.5)
   - Balanced stimulus presentation (equal left/right coherences per batch)
   - Curriculum gating: don't advance to history phases until p(right) ∈ [0.4, 0.6]

4. **The Decoupling Problem remains open.** Future work should first solve the bias problem, then re-evaluate per-trial history loss (and other mechanisms) in a regime where both psychometric and history metrics are operative.

---

## Phase 4: The Bias Was Never Real — Infrastructure Discovery (February 2026)

### The Mystery

All Hybrid experiments (E–H) and the bias-fix experiments (I–J) reported `p_right_overall ≈ 0.16`, interpreted as an "84% leftward bias." Multiple architectural fixes were attempted (exact DDM P(right) formula with starting-point bias, per-trial history loss), none of which moved the metric. This led to the suspicion that the number itself was wrong.

### Root Cause: Action Distribution Analysis

Direct counting of actions across runs G/H/I/J revealed:

| Run | Left | Right | Hold | Total | Commit Rate | p_right (all) | p_right (committed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G: Control | 2133 | 1973 | 7894 | 12000 | **34.2%** | 0.164 | **0.481** |
| H: Treatment | 2193 | 1979 | 7828 | 12000 | **34.8%** | 0.165 | **0.474** |
| I: Bias fix ctrl | 2133 | 1952 | 7915 | 12000 | **34.0%** | 0.163 | **0.478** |
| J: Bias fix treat | 2154 | 1945 | 7901 | 12000 | **34.2%** | 0.162 | **0.475** |

**The agent was never biased.** Among committed trials (left + right only), p_right ≈ 0.48 — nearly balanced. The "bias" was an artifact of `p_right_overall = right / total`, which includes hold/timeout trials in the denominator. With a 66% hold rate: `p_right_overall ≈ 0.34 × 0.48 ≈ 0.16`.

### Root Cause: Why 66% Hold Rate?

The DDM simulation (`_simulate_ddm()`) was configured with `max_commit_steps=200`, but the environment's response phase was only **120 steps** (1200ms). When the DDM planned a commit at step 121–200, the environment had already transitioned to the "outcome" phase, and the rollout code sent `ACTION_HOLD` during non-response phases. The env then called `_finalize_without_response()`, logging the trial as "hold".

The DDM itself **always** returns a committed choice (left or right) at timeout — it never produces a hold. The holds were entirely an artifact of the agent-environment timing mismatch.

### Fixes Applied

1. **`response_duration_override`**: The rollout now passes `max_commit_steps` to the env as `response_duration_override`, ensuring the response window matches the DDM's planning horizon.

2. **`effective_max_commit`**: A safety cap `min(config.max_commit_steps, response_phase.duration_steps)` prevents planning commits beyond the env window.

3. **`max_commit_steps` increased to 300**: With 120 steps, low-coherence DDM trials (mean ~95 steps, but with heavy right tail) frequently hit the ceiling, producing 1200ms RTs. With 300 steps (3 seconds), >95% of trials cross the boundary naturally.

4. **New metrics**: `p_right_committed` (right / (left+right)) and `commit_rate` ((left+right) / total) added to `eval/metrics.py` to distinguish committed choice bias from hold-rate effects.

5. **`rt_ok` threshold widened**: From 2000ms to 3500ms to accommodate the wider response window (DDM intercept at zero coherence naturally exceeds 2000ms when the accumulation process has 3 seconds to run).

### Impact

These were not tuning changes — they were **correctness fixes**. The previous experiments (E–H, I–J) were measuring hold-rate artifacts, not choice behaviour. All conclusions drawn from `p_right_overall ≈ 0.16` were invalid.

---

## Phase 5: Proper Decoupling Experiments — K2/L2 (February 2026)

With infrastructure bugs fixed, we re-ran the Decoupling experiment under correct conditions: 30 episodes, 400 trials/episode, full 7-phase curriculum, seed=42, max_commit_steps=300.

| Metric | K2: Control (per_trial=0.0) | L2: Treatment (per_trial=0.5) | Animal Target |
| --- | --- | --- | --- |
| **Commit rate** | **1.000** | **1.000** | — |
| **p_right_committed** | **0.496** | **0.495** | ~0.5 |
| **Psych slope** | **10.7** | **10.6** | 10–20 |
| **Psych bias** | **0.007** | **0.005** | ≈0 |
| **Chrono slope** | **−270 ms/unit** | **−264 ms/unit** | −100 to −645 |
| **RT range** | **1350 ms** | **1300 ms** | 200–400 |
| **Ceiling fraction** | **0.17** | **0.17** | 0.0 |
| Win-stay | 0.486 | 0.498 | 0.6–0.8 |
| Lose-shift | 0.506 | 0.477 | 0.3–0.5 |
| Sticky choice | 0.488 | 0.504 | 0.5–0.7 |
| prev_correct_beta | 0.041 | null | >0 |

### Interpretation

**Psychometric and chronometric are now excellent — simultaneously.**

This is a significant improvement relative to all prior runs:

1. **Psychometric slope of 10.7** — within the animal range (10–20) for the first time, with near-zero bias. The agent correctly discriminates motion direction across all coherence levels: p(right) goes from 0.002 at coh=−0.512 to 0.997 at coh=+0.512.

2. **Chronometric slope of −270 ms/unit** — appropriately negative, with RTs ranging from 890ms (high coherence) to 2150ms (zero coherence). The agent genuinely slows down for harder trials through DDM evidence accumulation, not ceiling clamping.

3. **100% commit rate** — every trial produces a left/right choice, eliminating hold-rate contamination.

4. **17% ceiling fraction** — only 1 of 6 coherence levels at ceiling, compared to 50%+ in prior runs.

**History metrics remain at chance.** Win-stay ≈ 0.49, lose-shift ≈ 0.49, repetition bias ≈ 0.0 — indistinguishable from a memoryless agent. The per-trial history loss (L2 vs K2) produced **no detectable effect** on any metric.

### Revised Understanding of the Decoupling Problem

With all infrastructure bugs fixed, the picture is now clear:

1. **Intra-trial dynamics are SOLVED.** The Hybrid DDM+LSTM with curriculum learning produces excellent psychometric and chronometric curves simultaneously when the agent-environment timing is correct.

2. **Inter-trial dynamics are NOT SOLVED.** No history loss variant (batch-mean, per-trial, or combined) produces above-chance win-stay or lose-shift in the rollout. The LSTM hidden state does not carry forward reward/choice information in a way that biases subsequent DDM parameters.

3. **The Decoupling Problem has narrowed.** It is no longer "agents can do one OR the other" — intra-trial dynamics are reliably captured. The remaining gap is purely in inter-trial history effects.

4. **Per-trial history loss is not the mechanism.** Despite theoretical motivation (Jensen's inequality), three experimental conditions (F, H, L2) at different scales showed no replicable benefit. The loss gradient exists (verified in unit tests) but does not translate to rollout-time history effects.

### Why History Fails: Architectural Hypothesis

The LSTM sets DDM parameters (drift_gain, bound, noise, bias, non_decision_ms) once at the start of each trial. These parameters fully determine the stochastic DDM simulation. Even if the LSTM's hidden state encodes trial history, the DDM parameter space may be too coarse to express subtle history biases:

- **Drift bias pathway**: History should modulate initial drift bias (starting point). But the bias head output (−0.02 to −0.10) is tiny relative to the bound (1.3), so its effect on choice probability is negligible.
- **Drift gain pathway**: History could modulate drift_gain to make the agent more/less sensitive after wins/losses. But drift_gain (≈30) is dominated by the coherence signal, so small history-driven modulations are washed out.
- **The DDM is too powerful**: With drift_gain=30 and bound=1.3, even zero-coherence trials cross the boundary purely by noise within ~95 steps. The DDM makes well-calibrated decisions regardless of history encoding.

Potential architectural solutions:

1. **Increase bias head range**: Scale bias output to ±0.5 instead of ±0.1, giving history a stronger lever on choice probability.
2. **Add a prior/lapse pathway**: A separate head for "prior probability of right" that combines with DDM output, rather than folding history through DDM parameters.
3. **Dynamic drift modulation**: Use the R-DDM approach where LSTM modulates drift at each timestep, not just at trial start.
4. **Reward prediction error**: Add an auxiliary RPE signal that directly modulates choice bias, bypassing the DDM parameter bottleneck.

---

## Phase 6: History Bias Head — Gradient Isolation Experiments (February 2026)

Following the K2 breakthrough, we pursued **Hypothesis 1** from above: give the model a dedicated history pathway with gradient isolation from the WFPT loss. The goal was to test whether the LSTM hidden state can drive history effects when freed from the gradient conflict with WFPT.

### Architecture: History Bias Head with Gradient Isolation

Added a dedicated `history_bias_head` (nn.Linear, hidden_size → 1) to the HybridDDMModel, zero-initialized so it starts with no effect. Gradient isolation ensures WFPT loss cannot flow to this head. During rollout, the history bias shifts the DDM starting evidence: `starting_point = bias + history_bias * scale * bound`. During training, the per-trial history loss provides gradient exclusively to the history_bias_head.

Nine gradient isolation tests verify the architecture is correct.

### Experiments

| Run | per_trial_history weight | history_bias_scale | Phase 7 epochs | Freeze | LR | win_stay | lose_shift | hb_weight max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `history_bias_head_control` | 0.0 (control) | 0.5 | 5 | No | 3e-4 | 0.486 | 0.496 | 0.000 |
| `history_bias_head_v2` | 0.5 | 0.5 | 5 | No | 3e-4 | 0.502 | 0.488 | 0.002 |
| `history_bias_head_v3_strong` | 2.0 | 1.0 | 20 | No | 3e-4 | 0.491 | 0.497 | 0.001 |
| `history_bias_head_v4_frozen` | 0.5 | 0.5 | 20 | Yes | 3e-3 | 0.485 | 0.503 | 0.004 |
| `history_bias_sigmoid_v5` | 0.5 | 0.5 | 20 | Yes | 3e-3 | 0.491 | 0.493 | 0.003 |
| `history_bias_pstay_v6` | 1.0 | 1.0 | 30 | Yes | 1e-2 | 0.493 | 0.503 | 0.008 |

All runs preserved K2-level intra-trial performance (psychometric slope 10.1–10.5, chronometric slope −268 to −297 ms/unit, 100% commit rate). **No run moved history metrics above chance.**

### Three-Layer Gradient Problem (Root Cause Analysis)

Systematic debugging revealed three stacked gradient pathologies that prevent the history_bias_head from learning:

**1. DDM choice probability saturation.** The analytical DDM formula used for the per-trial history loss saturates at high coherence, zeroing the gradient for ~5/6 of trials:

| Coherence | dp/d(hb_weight) |
| --- | --- |
| 0.000 | 0.210 |
| 0.032 | 0.160 |
| 0.064 | 0.076 |
| 0.128 | 0.010 |
| 0.256 | 0.000087 |
| 0.512 | 0.000000 |

**Fix (v5):** Replaced DDM formula with a simple sigmoid for the training loss path. Confirmed all coherences produce gradient (0.4–5.1). Rollout still uses the actual DDM simulation.

**2. Left/right gradient cancellation.** The original parameterization represents P(right). After right-wins, gradient pushes the head positive; after left-wins, negative. With roughly balanced actions, these anti-align (cosine similarity = −0.48) producing **72% gradient cancellation** within each training batch.

**Fix (v6):** Re-parameterized as P(stay) — all win trials push gradient in the same direction. Cancellation dropped to 56% (cosine similarity = +0.27).

**3. Optimization instability from session-dependent hidden states.** Even with fixes 1 and 2, the history_bias_head weights oscillate instead of converging. Direct tracing showed: step 0 → weight 0.01, step 10 → weight 0.0002, step 20 → 0.004, step 30 → 0.003. Different sessions push weights in different directions because:

- The LSTM hidden state has 64 dimensions — the history signal is encoded in session-dependent, arbitrary directions
- A linear head trained for ~30 epochs cannot find a consistent mapping from high-dimensional hidden states to a scalar history bias
- The LSTM was trained by WFPT/choice losses to encode *within-trial* dynamics; history information exists as a side effect in arbitrary dimensions

**This is the fundamental issue:** We verified the LSTM hidden state *does* encode history (mean absolute difference = 0.194 between post-win and post-loss states; drift_gain differs by 5.7). The information is there, but a small linear head cannot reliably extract it from the high-dimensional space in the training budget available.

### Scientific Conclusions

1. **Gradient isolation works correctly** — verified by 9 unit tests. WFPT loss cannot reach the history_bias_head.

2. **The LSTM encodes history but in a way that's not linearly extractable.** The hidden state representation is optimized for predicting DDM parameters, not for history bias extraction. The history signal is distributed across many dimensions in session-dependent patterns.

3. **The DDM parameter bottleneck is real.** Even when we provide a clean gradient path, the DDM choice probability formula creates a severe nonlinearity that makes learning slow at best and impossible at worst.

4. **The Decoupling Problem is not just a loss weighting issue.** We tested weights from 0.0 to 2.0, learning rates from 3e-4 to 1e-2, scales from 0.5 to 1.0, with and without freezing, with three different gradient formulations. The consistent failure rules out simple hyperparameter tuning as the solution.

5. **An architectural change is needed.** History processing likely requires a separate input pathway (not extracted from the LSTM hidden state) or a fundamentally different integration mechanism. This aligns with neuroscience models where trial-history biases originate in prefrontal/basal ganglia circuits separate from the sensory evidence accumulation pathway in parietal cortex.

### Architectural Implications and Design Choice

The Phase 6 experiments rule out the "post-hoc linear readout" approach to history. Three candidate architectures were considered:

1. **Separate history stream**: A dedicated pathway that processes (prev_action, prev_reward) directly into a starting-point bias, bypassing the LSTM hidden state entirely.
2. **Dynamic R-DDM**: LSTM modulates drift at every timestep within a trial, allowing history to influence the accumulation process dynamically.
3. **Prior mixture pathway**: A separate network computes P(right|history) which is mixed with the DDM choice via a learned gate.

**We chose Option 1 (Separate History Stream)** for the following reasons:

**Scientific rationale.** The approach directly tests a specific neuroscience hypothesis: that intra-trial dynamics (evidence accumulation) and inter-trial dynamics (history-dependent biases) arise from *anatomically and computationally distinct circuits* that converge at the level of starting-point bias.

This maps onto known primate decision-making circuitry:

| Brain circuit | Model component | Function |
| --- | --- | --- |
| LIP (lateral intraparietal area) | LSTM → DDM simulation | Sensory evidence accumulation |
| PFC (prefrontal cortex) + Basal ganglia | History network (prev_action, prev_reward → stay_tendency) | Recent outcome tracking, action value updating |
| PFC→LIP projections (baseline firing shift) | stay_tendency * scale * bound → starting point | History bias shifts accumulation starting point |
| Dopamine reward signals | Gradient from per-trial history loss | Learning signal for history circuit |
| Sensory statistics | Gradient from WFPT/choice loss | Learning signal for accumulation circuit |

This biological mapping also explains why the LSTM-readout approach (Phase 6) failed: we were asking the evidence accumulation circuit (LIP/LSTM) to both accumulate evidence AND compute history biases. In the brain, these are distinct circuits with distinct learning rules — the basal ganglia learns from dopamine reward signals while LIP learns from sensory evidence. Routing both learning signals through one circuit (the LSTM) creates the gradient conflict we observed.

**Practical rationale over alternatives:**

- **vs. Dynamic R-DDM (Option 2)**: The R-DDM fundamentally changes how evidence accumulation works, risking the intra-trial performance that took 55+ experiments to achieve. Option 1 is *additive* — it only adds a pathway, never changes what already works. If Option 1 fails, we learn that starting-point bias is insufficient for history effects (pushing toward Option 2). If we start with Option 2 and it fails, we learn nothing about why.

- **vs. Prior mixture (Option 3)**: A prior mixture adjusts the final choice probability *after* the DDM decision — a statistical correction, not a mechanistic one. Starting-point bias (Option 1) changes the entire trajectory of evidence accumulation, producing testable predictions about RT distributions: responses should be *faster* in the biased direction and *slower* in the opposite direction. This matches what's observed in animal data and provides a richer behavioral fingerprint to validate against.

**Architecture.** A small MLP takes (prev_action, prev_reward) as direct features — not extracted from the LSTM hidden state. It outputs a "stay tendency" (how much to repeat the previous action) that shifts the DDM starting point. The network *learns* the history rule from data rather than having it hardcoded. Zero-initialized output layer ensures no effect at the start of training, preserving all existing intra-trial performance through Phases 1–6.

### Phase 7 Result: The Reference Data Discovery

The separate history network (v7) implementation was architecturally sound — the network learned meaningful weights (output layer max 0.096 vs 0.003 for the LSTM-based head), and isolated gradient tests confirmed it converges in ~20 optimizer steps to correct win-stay/lose-shift probabilities.

However, rollout history metrics remained at chance (win_stay=0.492, lose_shift=0.517). Investigation revealed a fundamental issue with the **training targets**:

| Metric | Macaque reference (aggregate) | Per-session range | Our model |
| --- | --- | --- | --- |
| Win-stay | **0.458** | 0.222–0.548 | 0.492 |
| Lose-shift | 0.515 | 0.000–0.739 | 0.517 |

**The Roitman & Shadlen macaque does not show above-chance win-stay.** The aggregate win-stay is 0.458 (below 0.5), and only 7 of 27 sessions show win-stay > 0.5. This is consistent with an overtrained monkey that has learned to ignore previous outcomes and respond purely to current sensory evidence.

**Our model at win_stay=0.492 already matches the reference data better than any win-stay > 0.5 target would.** The per-trial history loss was trying to push the model toward a pattern that doesn't exist in the reference data.

### Methodological Failure: We Didn't Check the Reference Data

The most important lesson from this entire effort is embarrassingly simple: **we never checked whether the reference data exhibited the history effects we were trying to capture.**

The FINDINGS.md has stated "win-stay ≈ 0.49" for the model since the K2 run and framed it as a failure — the "Decoupling Problem." But at no point did anyone compare the model's win-stay to the reference animal's win-stay. We assumed "animals have history effects" from the general neuroscience literature without verifying it against the Roitman & Shadlen dataset specifically.

The Roitman & Shadlen macaque was heavily overtrained (2,611 published trials of the same motion discrimination task). History effects are known to diminish with overtraining as the animal learns to ignore previous outcomes and respond purely to current sensory evidence. This is well-established in the literature, and we should have anticipated it.

Six experiments, three architectural fixes, and extensive gradient debugging were spent optimizing toward a target that doesn't exist in the data. The architecture is mechanically sound but **untested against data with genuine history effects**.

### What Was Actually Learned

1. **Three gradient pathologies are real.** DDM formula saturation (only 1/6 of trials produce gradient at high coherence), left/right gradient cancellation (72% cancellation in P(right) parameterization), and optimization instability from session-dependent LSTM hidden states. These are genuine obstacles that would block any LSTM-readout approach to history, regardless of dataset. The diagnostic methodology for discovering them is reusable.

2. **The separate history stream architecture is the correct approach.** A small MLP that takes (prev_action, prev_reward) directly, bypassing the LSTM, avoids all three gradient pathologies. Isolated gradient tests confirm it converges to correct win-stay/lose-shift targets in ~20 optimizer steps. But it has not been validated on data with genuine history effects.

3. **The Decoupling Problem, as originally stated, was partly an artifact.** For the Roitman & Shadlen macaque, intra-trial dynamics and inter-trial dynamics are not "decoupled" in the model — they are both correctly captured. The model's chance-level history matches the reference animal's chance-level history. The Decoupling Problem remains real for datasets with genuine history effects (e.g., IBL mouse: win-stay 0.73, lose-shift 0.34).

4. **Would more macaque data help?** Potentially — data from less-trained animals, or from paradigms where history effects are more prominent, would provide the training signal the architecture needs. The overtraining explanation predicts a specific, testable gradient: early-session data should show stronger history effects than late-session data.

### Next Steps

The IBL mouse dataset (win-stay 0.73, lose-shift 0.34) provides a clear, strong history signal. The Hybrid DDM+LSTM with the separate history stream should be adapted to the IBL 2AFC task. This is the proper test of whether the architecture can capture inter-trial dynamics when the reference data actually exhibits them.

---

## Phase 8: IBL Mouse Adaptation — Drift-Rate Bias Solves the Decoupling Problem (February 2026)

### Motivation

The Phase 7 discovery that Roitman & Shadlen macaque data lacks history effects prompted a switch to IBL mouse data, which has strong history signals (win-stay=0.724, lose-shift=0.427, 8,406 trials, 10 sessions). This is the proper test of the separate history stream architecture.

### IBL 2AFC Adaptation

The Hybrid DDM+LSTM was parameterized by task (`--task ibl_2afc`) rather than forked. Changes:

- Task-conditional environment (IBL2AFCEnv vs RDMMacaqueEnv)
- Stimulus: signed contrast [-1, 1] vs signed coherence
- IBL-specific RT targets (652–2253 ms vs 464–785 ms for macaque)
- Block-prior-aware correct-answer logic for zero-contrast trials
- Extended response window to accommodate IBL RT range

The model architecture, training loop, loss functions, and history_network are identical across tasks — only data loading and rollout differ.

### Starting-Point Bias Experiments (v1–v3)

| Run | history_bias_scale | history_drift_scale | Phase 7 epochs | per_trial_history | Win-stay | Lose-shift | Chrono slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ibl_hybrid_curriculum` (v1) | 0.5 | 0.0 | 5 | 0.5 | 0.547 | 0.525 | -73.1 |
| `ibl_history_v2_long` | 0.5 | 0.0 | 20 | 0.5 | 0.544 | 0.530 | -77.0 |
| `ibl_history_v3_strong` | 1.0 | 0.0 | 20 | 1.5 | 0.544 | 0.533 | -77.3 |
| IBL mouse reference | — | — | — | — | **0.724** | **0.427** | negative |

Win-stay plateaued at ~0.544 regardless of training duration or loss weight. Diagnostic analysis revealed the trained history_network outputs correct values — with scale=1.0, the sigmoid P(stay) after wins was 0.725, nearly exactly matching the target. But rollout win-stay was only 0.544.

### Root Cause: Starting-Point Bias Is the Wrong Mechanism

DDM simulation showed that starting-point bias only affects choice on ambiguous trials:

| |Contrast| | P(right) shift from starting-point bias |
| --- | --- |
| 0.0 | +11.5% |
| 0.0625 | +5.1% |
| 0.125 | +1.2% |
| 0.25 | +0.1% |
| 1.0 | +0.0% |

At |contrast| >= 0.125, the drift rate (drift_gain * contrast ≈ 14 * 0.125 = 1.75) overwhelms any starting-point shift. Since ~60% of trials have |contrast| >= 0.125, the starting-point mechanism can only move overall P(stay) by ~2-4 percentage points.

**Real mice show history effects even on easy trials.** This means history must affect the evidence accumulation process itself, not just the starting point.

### The Fix: Drift-Rate Bias

Added a `history_drift_scale` parameter. During rollout, the history_network's stay_tendency now adds a drift-rate bias in addition to starting-point bias:

```
history_drift = stay_tendency * history_drift_scale * prev_direction
effective_drift = drift_gain * stimulus + history_drift
```

This affects ALL trials — even at high coherence, a drift-rate bias subtly shifts the accumulation trajectory, influencing both choice and RT.

### Drift-Rate Bias Experiments (v4–v6)

| Run | history_drift_scale | Win-stay | Lose-shift | Sticky choice | Psych slope | Chrono slope |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (baseline, no drift) | 0.0 | 0.547 | 0.525 | 0.530 | 6.70 | -73.1 |
| v4 (moderate drift) | 5.0 | 0.585 | 0.487 | 0.569 | 7.28 | -72.2 |
| v5 (strong drift) | 8.0 | 0.607 | 0.458 | 0.592 | 6.61 | -72.5 |
| **v6 (max drift)** | **15.0** | **0.655** | **0.402** | **0.642** | 6.04 | -66.6 |
| IBL mouse reference | — | **0.724** | **0.427** | **0.692** | ~13.2 | negative |

All drift-rate experiments preserved: 100% commit rate, negative chronometric slopes, psychometric discrimination.

### Scientific Implications

**1. The Decoupling Problem is architecturally solved.** For the first time in 60+ experiments, we have an agent that simultaneously produces all three behavioral fingerprints: negative chronometric slope (intra-trial dynamics), above-chance history effects (inter-trial dynamics), and psychometric discrimination (accuracy scales with evidence). This is achieved via the Attention-Gated History Bias mechanism. Quantitative calibration (matching the exact psychometric slope, lapse rates, and history asymmetry) remains ongoing — see Multi-Seed Validation below for the current state.

**2. History effects require drift-rate bias, not just starting-point bias.** Starting-point bias (the standard neuroscience model for history-dependent DDM) only affects ambiguous trials. Drift-rate bias affects all trials, matching the empirical observation that mice show win-stay even on easy discriminations.

This has a neuroscience interpretation: history doesn't just set the "ready position" before evidence arrives (starting point) — it continuously biases how evidence is *processed* throughout the trial (drift rate). In neural terms, this suggests PFC/basal ganglia history signals project not only to LIP baseline firing rates but also modulate the gain of sensory evidence during accumulation.

**3. The separate history stream architecture works.** The MLP that takes (prev_action, prev_reward) directly, bypassing the LSTM, successfully learns history patterns and translates them into behavioral effects. This validates the biological hypothesis that history processing and evidence accumulation are computationally distinct circuits.

**4. A tradeoff between history and discrimination may exist.** Psychometric slope decreases slightly with stronger history drift (6.70 → 6.04 as drift_scale increases). This mirrors animal behavior — mice with stronger history biases tend to be slightly less accurate on the current trial. The tradeoff suggests that history bias literally interferes with evidence accumulation, which is exactly what drift-rate bias does mechanistically.

### Phase 9: The Joint Learning Reality Check (February 2026)

All previous experiments (v1-v6) contained a major methodological flaw: the `history_drift_scale` and `history_bias_scale` were **hardcoded constants** (15.0 and 1.0 respectively). They were manually injected into the DDM during rollout and likelihood estimation, but the history network itself was trained via an entirely separate, disjoint likelihood objective (`cross_entropy` for stay_tendency). 

The gradient from the DDM's `wfpt_loss` naturally reflecting how history *should* interact with evidence accumulation **never flowed back into the history network**. The "success" of the v6 experiment was artificially manufactured by a 15.0 regularization constant that forced the model to behave correctly.

#### The Joint Learning Fix
We refactored `HybridDDMModel` to make `history_bias_scale` and `history_drift_scale` true learnable `nn.Parameter` tensors. We then fundamentally rewired the training loop so that `effective_drift` and `effective_bias` (which include the history terms) are directly evaluated by the `wfpt_loss`, establishing a single, united computation graph. 

We ran a 5-seed validation sweep (`runs/seed_sweep_v6_joint`) using the exact `v6_max` curriculum configuration (30 episodes, 80 max sessions, 20 history epochs) to test whether the agent could *organically* learn the history scales.

| Seed | Win-stay | Lose-shift | Sticky choice | Psych slope |
| --- | --- | --- | --- | --- |
| 42 | 0.8792 | 0.1382 | 0.8737 | 2.6339 |
| 123 | 0.9138 | 0.1036 | 0.9078 | 2.0918 |
| 256 | 0.9228 | 0.1040 | 0.9136 | 2.0488 |
| 789 | 0.9133 | 0.1209 | 0.9020 | 2.1266 |
| 1337 | 0.8985 | 0.1443 | 0.8850 | 2.4386 |
| **Mean** | **0.9055** | **0.1222** | **0.8964** | **2.2679** |
| Target | 0.7240 | 0.4270 | — | 13.2000 |

#### The Mode Collapse Reality
The 5-seed sweep reveals a harsh truth about the mathematical optimization: **extreme fragility and mode collapse**. 

Without the safety net of the hardcoded `15.0` heuristic enforcing correct history scaling, *all 5 seeds* learned that the easiest way to minimize the loss is to **almost entirely ignore the sensory stimulus** (evident in the crashed `psych_slope` of ~2.2 compared to the target 13.2) and simply hit the same button repeatedly (evident in the `win_stay` near 0.90 and `lose_shift` near 0.12). 

The gradients properly flowed, but they found a degenerate local optimum. The baseline finding that the agent can reliably disentangle evidence accumulation from history *without* heavy regularization is officially false.

### Recommended Path Forward: Biologically Plausible Mechanisms
To solve the Decoupling Problem mathematically, we must understand how biological brains avoid this mode collapse. Real animals don't stare at a high-contrast grating and ignore it because they're historically biased. We recommend exploring three specific mechanisms:

1. **Sensory Evidence Obligation (Structural Bounds):** Real brains process strong sensory stimuli obligatorily in V1. We could enforce a lower bound on the sensory drift weight, or structurally limit the maximum magnitude of the history bias relative to the sensory input, forcing the agent to always "see" the stimulus.
2. **Asymmetric/Attention-Gated History Bias:** The current model constantly pushes the DDM with history drift throughout the entire trial. In biology, history may act more as a "prior" that only dominates when confidence is low or in the first few milliseconds. We could scale the history bias inversely proportionally to the stimulus strength.
3. **Reward Prediction Error (RPE) Separation:** Dopamine pathways drive win-stay behavior via RPE, distinct from DDM likelihood matching. Giving the history network an auxiliary TD/Q-learning loss, rather than forcing it to optimize purely for reaction-time likelihood matching, might provide the biologically grounded representation needed to stop it from collapsing the main DDM pathway.

#### Hypothesis 2 Validation: Attention-Gated History Bias
To test the second hypothesis, we implemented an attention gate on the `history_drift` parameter dynamically during training and rollout. We defined sensory `confidence` as `min(abs(stimulus), 1.0)`, and scaled the history drift proportionally to `(1.0 - confidence)`.

This mechanical suppression forces the agent to rely entirely on the sensory stimulus when the grating is high-contrast, preventing the history network from discovering the cheat code of ignoring the stimulus. 

To formally validate this, we ran the standard 5-seed validation suite on the attention-gated version of the joint-learning problem:

**Results from 5-Seed Attention-Gated Validation:**
| Seed | Win-stay | Lose-shift | Sticky choice | Psych slope |
| --- | --- | --- | --- | --- |
| 42 | 0.8174 | 0.2253 | 0.8063 | 4.2592 |
| 123 | 0.8489 | 0.1822 | 0.8401 | 3.7434 |
| 256 | 0.8491 | 0.1871 | 0.8393 | 3.6543 |
| 789 | 0.8478 | 0.1930 | 0.8366 | 3.7164 |
| 1337 | 0.8343 | 0.2408 | 0.8149 | 4.1748 |
| **Mean** | **0.8395** | **0.2057** | **0.8274** | **3.9096** |
| Target | 0.7240 | 0.4270 | — | 13.2000 |

**Conclusion:** The attention gate mathematically shatters the `v6_joint` mode collapse across all random seeds. The psychometric slope rebounded from the collapsed ~2.26 up to a very stable ~3.91, and win-stay reduced from the degenerate ~0.90 down to an average of ~0.84. 

While there is still work to be done to perfectly shape the chronometric slopes and increase the sharpness of the psychometric curve (the psychometric slope is still shallower than the mouse target of 13.2), the agent is now demonstrably capable of balancing history priors with sensory evidence dynamically without collapsing. This is a tremendous step forward for the Decoupling Problem, proving a biological mechanism is required for stable joint learning.

### Conceptual Implications

The success of the Attention-Gated History Bias mechanism has several profound implications for computational neuroscience and the goals of AnimalTaskSim:

1. **The "Decoupling Problem" is a Feature of Unconstrained Joint Learning, Not a Bug:** Before this fix, when the agent was allowed to learn both sensory evidence accumulation and history biases simultaneously, it suffered from severe mode collapse. It learned that repeating the previous action is mathematically "easier" than interpreting ambiguous visual gratings. Because both signals were combined linearly, gradient descent took the lazy path: it maximized the history bias weight and ignored the stimulus. This implies that in artificial networks (and likely in early biological learning), jointly training history priors and sensory evidence accumulation is inherently unstable without structural guardrails.

2. **Biology Uses Top-Down Attentional Gating:** The formula `gated_history_drift = history_drift * (1.0 - confidence)` acts as a biological structural guardrail. It implies that animals do not blindly add internal priors to external evidence. Instead, a top-down attentional gating mechanism (likely managed by the prefrontal cortex or basal ganglia) dynamically suppresses priors based on stimulus clarity. When the stimulus is ambiguous (0% contrast), the gate opens fully, and the animal relies on its history prior. When the stimulus is obvious (100% contrast), the gate snaps shut, suppressing the history prior so the decision is driven entirely by objective sensory evidence.

3. **Artificial Heuristics Cannot Replace Biological Fidelity:** Previous attempts to prevent mode collapse relied on disjointed training—training the visual network first, freezing it, and then training the history network. While this artificial heuristic works for maximizing reward or matching metrics downstream, it is biologically implausible, as animals learn continuously. The 5-seed sweep proves that stable, joint, continuous learning is achievable by hardcoding biological constraints into the architecture rather than relying on artificial curriculum tricks.

This implies the model is now fundamentally more robust and brain-like. To prevent an intelligent agent from falling into degenerate loops (like exploiting a win-stay strategy), it requires an internal attentional gauge that mathematically down-weights its priors when confronted with strong, objective reality.


---

## Phase 10: Differentiable DDM Simulator and Psychometric Slope Breakthrough (February 2026)

### The Mathematical Exploit

When attempting to steepen the psychometric slope (from ~3.9 toward the IBL target of 13.2) by increasing choice loss weight in the curriculum, the agent discovered a mathematical loophole in the analytical DDM training equations.

The training loop used the standard expected RT formula:

```
E[RT] = (A/v) * tanh(v*A/σ²)
```

Under high choice loss pressure, the LSTM learned to push the decision bound `A → ∞` while simultaneously crushing the drift rate `v → 0`. In this limit, `tanh(v*A/σ²) → 0`, which collapsed the gradient of the RT penalty with respect to the bound. The agent preferred eating a flat 3000ms timeout penalty (from the environment ceiling) over facing the steep gradients of the BCE choice loss. Every trial timed out — 100% degenerate.

### The Fix: Differentiable Euler-Maruyama Simulation

The analytical approximations are fundamentally unsuited for backpropagation in environments with hard max-step cutoffs. The fix replaced the analytical `_ddm_choice_prob()` and `mean_steps` formulas with a **differentiable DDM simulator** that unrolls the actual stochastic evidence accumulation as a PyTorch tensor operation:

```
ΔE = v·Δt + σ·√Δt·N(0,1)
```

Key implementation details:

1. **Evidence trajectory**: `evidence = bias + cumsum(drift*dt + noise*sqrt_dt*randn)` over `max_commit_steps` (120 steps = 3000ms)
2. **Soft boundary crossing**: Sigmoid activation `σ((evidence - bound) / temp)` instead of hard threshold, preserving gradient flow
3. **Commit density**: `P(commit at step t) = P(cross at t) * ∏_{s<t}(1 - P(cross at s))` — a proper first-passage density via cumulative product
4. **Expected RT**: `E[RT] = Σ(t * commit_density_t) * step_ms + non_decision_ms`
5. **Timeout penalty**: `P(timeout) * max_steps * 10` — massive gradient if the agent never crosses the bound
6. **Choice probability**: Ratio of upper-bound crossings to total crossings, weighted by the commit density

The agent can no longer hide behind the `tanh(0)` asymptote. At every simulated timestep, PyTorch traces the gradient from the RT penalty back through the bound parameter. If the agent inflates the bound, it feels a non-zero gradient at each of the 120 steps pushing it back down.

### Sweep Design: drift_scale × choice_weight

A 3×3 sweep over `drift_scale ∈ {10, 20, 30}` × `choice_weight ∈ {0.5, 1.0, 1.5}` with a 3-phase curriculum:

| Phase | Epochs | Choice | RT | History | Drift Magnitude |
|-------|--------|--------|-----|---------|-----------------|
| 1: RT only | 15 | 0.0 | 1.0 | 0.0 | 0.5 |
| 2: Add choice | 10 | cw×0.5 | 0.8 | 0.1 | 0.5 |
| 3: Full balance | 10 | cw | 0.5 | 0.2 | 0.5 |

This curriculum removes WFPT loss entirely (which was part of the old 7-phase curriculum) and uses drift magnitude regularization instead, with graduated choice pressure across phases.

### Results

| drift | choice_w | Psych Slope | Chrono Slope | Win-Stay | Lose-Shift | Bias | Quality |
|-------|----------|-------------|--------------|----------|------------|------|---------|
| 10 | 0.5 | 5.25 | ~0 | 0.718 | 0.167 | -0.155 | **degenerate** |
| 10 | 1.0 | **14.17** | -288 | 0.657 | 0.196 | -0.026 | all pass |
| 10 | 1.5 | **14.40** | -285 | 0.659 | 0.188 | -0.026 | all pass |
| **20** | **0.5** | **13.78** | **-286** | **0.654** | **0.211** | **-0.023** | **all pass** |
| 20 | 1.0 | 22.25 | -67 | 0.554 | 0.553 | 0.002 | all pass |
| 20 | 1.5 | 23.73 | -66 | 0.548 | 0.551 | 0.002 | all pass |
| 30 | 0.5 | 23.32 | -65 | 0.545 | 0.573 | 0.001 | all pass |
| 30 | 1.0 | 22.05 | -63 | 0.550 | 0.563 | 0.004 | all pass |
| 30 | 1.5 | 24.08 | -65 | 0.550 | 0.564 | 0.001 | all pass |

**IBL mouse targets:** psych slope ~13.2 | win-stay 0.724 | lose-shift 0.427 | chrono slope negative | bias ~0

### Three Regimes

The results reveal three distinct behavioral regimes:

**1. Degenerate (drift=10, choice=0.5):** Insufficient choice pressure. Despite the differentiable simulator, the agent still times out at 3000ms on every trial because the choice loss is too weak to create accuracy pressure. The simulator fix is necessary but not sufficient — it must be paired with adequate choice weight.

**2. Sweet spot (drift=10/choice≥1.0 and drift=20/choice=0.5):** Psychometric slopes of 13.8–14.4, chronometric slopes ranging from -65 to -286 ms/unit, win-stay ~0.555–0.655, all quality flags passing. *(See multi-seed validation caveat below.)*

**3. Over-discriminating (drift≥20/choice≥1.0 and all drift=30):** Psychometric slopes overshoot to 22–24 (~2× the target). History effects collapse to chance (win-stay ~0.55). Chrono slopes weaken to ~-65 ms/unit. The agent becomes too accurate and stops relying on history.

### The History-Accuracy Tradeoff (Quantified)

The sweep provides the clearest evidence yet of the tradeoff between perceptual discrimination and history dependence:

| Psych Slope Range | Win-Stay Range | Interpretation |
|-------------------|----------------|----------------|
| 5–6 (too shallow) | 0.66–0.72 | Agent guesses often → history dominates |
| **13–15 (target)** | **0.65–0.66** | **Balanced: evidence + history both active** |
| 22–24 (too steep) | 0.54–0.55 | Agent over-relies on evidence → history irrelevant |

This mirrors biological data: mice with stronger history biases tend to be slightly less accurate on the current trial, because the drift-rate bias literally interferes with evidence accumulation.

### Scientific Significance

**1. Analytical DDM gradients are unreliable for bounded environments.** The `tanh(κ)` exploit demonstrates that closed-form DDM equations, while mathematically correct for infinite-horizon problems, create degenerate gradient landscapes when the environment imposes hard time limits. Differentiable simulation is the correct approach for training DDM-based agents in bounded environments.

**2. The three-regime structure is a prediction.** The sweep predicts that biological decision-makers should fall in a specific band of the psych_slope vs win-stay space. Animals with very steep psychometric curves (high sensitivity) should show weak history effects, and vice versa. This is testable across individual mice in the IBL dataset.

**3. The architecture is correct; calibration is the remaining challenge.** The differentiable DDM + history MLP + attention gate architecture simultaneously produces all three behavioral fingerprints (psychometric discrimination, negative chronometric slope, above-chance history effects). The remaining work is quantitative — finding the drift_scale that matches the target psychometric slope without introducing ceiling artifacts.

### Multi-Seed Validation (February 2026)

The sweep winner (drift=20, choice=0.5) was validated across 5 seeds (42, 123, 256, 789, 1337). **The single-seed result of psych=13.78 did not replicate.** Investigation revealed it was an artifact of RT ceiling saturation that was subsequently fixed by code improvements to the hybrid trainer.

#### Original single-seed result (pre-refactor)

| Metric | Value | Notes |
|--------|-------|-------|
| Psych slope | 13.78 | Driven by 28% lapse rate |
| Chrono slope | -286 ms/unit | Driven by ceiling step-function (2/6 RT levels at 3000ms) |
| Win-stay | 0.654 | |
| Lapse low | 0.280 | 5.6× the animal target |

The RT profile was `3000/3000/2615/1180/580/340` ms across contrast levels — a step function between ceiling-clamped and non-clamped levels, not the smooth gradient seen in animal data.

#### Multi-seed result (post-refactor, same config)

| Metric | Mean ± SD | IBL Target | Verdict |
|--------|-----------|------------|---------|
| Psych slope | 22.96 ± 1.94 | 13.2 | 74% overshoot |
| Chrono slope | -64.1 ± 2.1 ms/unit | negative | Qualitatively correct, smooth RT gradient |
| Win-stay | 0.565 ± 0.009 | 0.724 | Below target |
| Lose-shift | 0.551 ± 0.030 | 0.427 | Above target (agent over-shifts) |
| Lapse | ~0.002 | ~0.05 | Near-zero (agent too accurate) |
| Bias | -0.004 ± 0.010 | ~0 | Excellent |
| Commit rate | 100% | ~100% | Match |

The RT profile is now smooth (`910/850/670/480/330/250` ms), range 660ms, with no ceiling saturation. This is a genuinely healthier result despite the psych slope overshoot — the agent's behavior is no longer confounded by environmental ceiling artifacts.

**Key insight**: The apparent 96% psych slope match was a coincidence of high lapse rates and RT ceiling saturation producing a flattened psychometric curve. The true psychometric sensitivity at drift_scale=20 is ~23, which is too steep. The curriculum is highly reproducible (SD=1.94 across seeds) but needs drift_scale reduction.

#### Follow-up: drift calibration sweep

A drift_scale sweep ({10, 12, 14} × 3 seeds) is in progress to find the value that produces psych slope ~13 without ceiling artifacts. Linear extrapolation suggests drift_scale ~11–12 should be in range.

### Drift Calibration Sweep v2 (Old Architecture)

A drift_scale sweep ({10, 12, 14} × 3 seeds) was completed using the original single-history-network architecture (no lapse, no asymmetric pathways).

| Drift | Psych Slope | Chrono Slope | Lapse Lo | Win-Stay | Lose-Shift |
|-------|------------|-------------|----------|----------|------------|
| 10 | 21.2 ± 1.8 | -67.1 ± 2.4 | ~0.00 | 0.56 ± 0.01 | 0.57 ± 0.01 |
| 12 | 21.0 ± 1.3 | -65.9 ± 1.3 | ~0.00 | 0.56 ± 0.01 | 0.56 ± 0.01 |
| 14 | 20.6 ± 2.1 | -67.4 ± 7.0 | ~0.03 | 0.55 ± 0.00 | 0.55 ± 0.01 |
| Target | 13.2 | negative | ~0.05 | 0.724 | 0.427 |

**Key finding:** Psych slope is insensitive to drift_scale in the old architecture. Reducing from 20 to 10 only moved slope from 23 to 21. History effects remain symmetric (WS ≈ LS ≈ 0.55-0.57) and lapse remains near zero at drift ≤ 12. These results motivated the architectural changes below.

### Asymmetric History Networks + Stochastic Lapse (February 2026)

Two biologically-motivated mechanisms were added to address the remaining gaps:

1. **Asymmetric history networks**: Replaced the single `history_network` MLP with separate `win_history_network` and `lose_history_network`. Routes through win pathway when `prev_reward > 0.5`, lose pathway otherwise. Models the dopaminergic asymmetry between reward and punishment processing observed in animal brains.

2. **Stochastic attention lapse**: On a fraction of trials the agent disengages and guesses randomly (P=0.5), producing the attentional lapses observed in animals (~5% in IBL mice).

#### Learnable Lapse Experiment (Negative Result)

The initial implementation used a **learnable** lapse parameter (`lapse_logit` as an `nn.Parameter`, trained via gradient descent through `prob_right = (1-sigmoid(logit))*prob + sigmoid(logit)*0.5`). This was the principled choice — let the model discover the appropriate lapse rate from the data.

**Single validation at drift=20** (seed=42) was promising:

| Metric | Before (old arch) | After (learnable lapse) | Animal Target |
|--------|-------------------|------------------------|---------------|
| Psych slope | 22.96 | 5.60 | 13.2 |
| Lapse low | ~0.002 | 0.042 | ~0.05 |
| Lapse high | ~0.002 | 0.107 | ~0.10 |
| Win-stay | 0.565 | 0.620 | 0.724 |
| Lose-shift | 0.551 | 0.414 | 0.427 |
| Chrono slope | -64.1 | -56.5 | negative |

The asymmetric history produced the correct WS > LS direction, and lose-shift was almost exactly at target.

**However, the sweep revealed optimizer exploitation.** At drift={25, 30, 35} × 3 seeds:

| Drift | Psych Slope | Lapse Lo | Lapse Hi | Win-Stay | Lose-Shift |
|-------|------------|----------|----------|----------|------------|
| 25 | 8.17 ± 0.41 | 0.149 ± 0.002 | 0.150 ± 0.005 | 0.549 ± 0.008 | 0.508 ± 0.007 |
| 30 | 7.56 ± 0.75 | 0.152 ± 0.010 | 0.152 ± 0.004 | 0.552 ± 0.007 | 0.506 ± 0.011 |
| 35 | 6.03 ± 0.27 | 0.143 ± 0.004 | 0.150 ± 0.001 | 0.539 ± 0.006 | 0.514 ± 0.009 |

The lapse rate tripled from ~5% (drift=20) to ~15% (drift=25+), becoming symmetric (lapse_lo ≈ lapse_hi ≈ 0.15). The optimizer exploited `lapse_logit` as a shortcut: higher lapse reduces choice loss on hard trials because guessing 50/50 on ambiguous stimuli is "less wrong" than a confident wrong answer. This is not what animal lapse represents — animal lapse is attentional disengagement, a hardware property of the brain's vigilance system, not a learned optimization strategy.

The history asymmetry also collapsed at higher drift scales (WS ≈ LS ≈ 0.55), likely because the high lapse rate drowns out the history signal.

**Decision: Fixed lapse rate.** `lapse_logit` was removed from the model. Lapse is now a fixed configuration parameter (`lapse_rate: float = 0.05`) applied as a Bernoulli gate in rollout only — `random() < lapse_rate` causes a random guess with random RT.

This respects the biological reality that attentional lapse is a fixed property of the animal's vigilance system, not an adaptive strategy. It also eliminates a confound — sweep results can now be interpreted cleanly without wondering whether lapse rate co-varied with drift scale.

#### Training Lapse — Double-Counting (Negative Result)

An intermediate implementation applied fixed lapse in both training (`prob_right = (1-lapse)*prob + lapse*0.5`) and rollout. A sweep at drift={20, 22, 25} × 3 seeds (`sweep_fixed_lapse_v1`) showed:

| Drift | Psych Slope | Chrono Slope | Win-Stay | Lose-Shift | Lapse Lo |
|-------|-------------|--------------|----------|------------|----------|
| 20 | 8.51 ± 0.24 | -38.1 ± 1.2 | 0.569 ± 0.005 | 0.512 ± 0.009 | 0.028 |
| 22 | 8.56 ± 0.17 | -37.2 ± 1.1 | 0.565 ± 0.005 | 0.520 ± 0.006 | 0.027 |
| 25 | 8.41 ± 0.30 | -35.3 ± 0.9 | 0.563 ± 0.007 | 0.517 ± 0.008 | 0.029 |

Psych slope was stuck at ~8.5 regardless of drift_scale — completely insensitive. The diagnosis: the reference animal data already contains the animal's own lapse. Blending additional lapse into the training probability double-counts it, compressing the dynamic range of choice gradients and flattening the psychometric curve. Training lapse was removed; lapse is applied only in rollout.

#### Rollout-Only Lapse Sweep

With lapse only in rollout, a sweep at drift={20, 22, 25} × 3 seeds (`sweep_rollout_lapse_v1`) showed:

| Drift | Psych Slope | Chrono Slope | Win-Stay | Lose-Shift | Lapse Lo |
|-------|-------------|--------------|----------|------------|----------|
| 20 | 9.57 ± 0.37 | -40.7 ± 1.8 | 0.566 ± 0.006 | 0.516 ± 0.012 | 0.026 |
| 22 | 9.20 ± 0.63 | -39.5 ± 0.8 | 0.566 ± 0.007 | 0.522 ± 0.003 | 0.024 |
| 25 | 8.96 ± 0.46 | -36.7 ± 0.2 | 0.565 ± 0.009 | 0.526 ± 0.010 | 0.027 |

Still too shallow (9.5 vs target 13.2) and still insensitive to drift. This led to the curriculum confound discovery below.

### The Curriculum Confound — Critical Discovery (February 2026)

All the new architecture sweeps (learnable lapse, fixed lapse, rollout-only lapse) used the **7-phase WFPT curriculum** (`--use-default-curriculum`). The old multi-seed validation (psych=22.96) used the **3-phase curriculum** (`--no-use-default-curriculum`). A controlled experiment isolated the variable.

#### The Experiment

New architecture (asymmetric history + rollout lapse) tested with the old 3-phase curriculum at drift=20, 3 seeds (`sweep_3phase_newarch`):

| Drift | Seed | Psych | Chrono | WS | LS | Lapse Lo |
|-------|------|-------|--------|------|------|----------|
| 20 | 42 | 20.57 | -69.3 | 0.555 | 0.560 | 0.020 |
| 20 | 123 | 16.36 | -60.0 | 0.563 | 0.568 | 0.016 |
| 20 | 256 | 18.08 | -61.1 | 0.550 | 0.532 | 0.021 |
| **AVG** | | **18.34 ± 2.12** | **-63.5 ± 5.1** | **0.556** | **0.553** | **0.019** |

#### The Verdict

| Configuration | Psych | Chrono | WS | LS | Lapse |
|---------------|-------|--------|------|------|-------|
| Old arch + 3-phase | 22.96 ± 1.94 | -64.1 ± 2.1 | 0.565 | 0.551 | ~0.002 |
| **New arch + 3-phase** | **18.34 ± 2.12** | **-63.5 ± 5.1** | **0.556** | **0.553** | **~0.019** |
| New arch + 7-phase | 9.57 ± 0.37 | -40.7 ± 1.8 | 0.566 | 0.516 | ~0.026 |
| **Animal target** | **13.2** | **negative** | **0.724** | **0.427** | **~0.05** |

**The 7-phase WFPT curriculum was responsible for the psych slope collapse (23 → 9.5), not the architecture change.** Switching the new architecture back to the 3-phase curriculum restored psych slope to 18.3 — still lower than the old architecture's 23 (the ~5-point reduction is from the rollout lapse flattening the psychometric curve), but now responsive to drift_scale and much closer to the 13.2 target.

#### Why WFPT Training Suppressed Psychometric Sensitivity

The 7-phase curriculum starts with a 15-epoch WFPT warmup phase where the *only* training signal is the Wiener First Passage Time likelihood — no choice loss at all. WFPT optimizes for the joint probability of choice and RT through the analytical DDM likelihood, and it depends on all DDM parameters simultaneously (drift, bound, noise, non-decision time).

When WFPT dominates training, the optimizer is free to adjust *any* parameter combination to maximize likelihood. It discovers that increasing noise and lowering bounds produces decent likelihood scores while reducing effective drift sensitivity. The model settles into a **high-noise, low-sensitivity regime** — RT distributions look statistically reasonable but the agent can barely discriminate stimuli (psych slope ~9).

The simpler 3-phase curriculum avoids this by:
1. Teaching RT structure first via direct MSE loss (no analytical likelihood)
2. Gradually layering choice accuracy on a stable RT foundation
3. Using drift magnitude regularization to anchor drift_gain scale

This mirrors how real brains develop — basic sensory circuits mature before higher-order decision circuits, establishing stable signal-to-noise ratios before complex optimization can distort them. WFPT warmup is like asking a developing brain to optimize complex statistical properties before basic edge detection is in place.

#### Implications for Next Steps

- The 3-phase curriculum is the correct training foundation for this architecture.
- ~~Reducing drift_scale from 20 to ~14-16 should bring psych slope from 18.3 to ~13.2.~~ See drift calibration below — `drift_scale` turned out to be a dead knob.
- History asymmetry (WS=0.556 vs 0.724, LS=0.553 vs 0.427) remains symmetric because the 3-phase curriculum does not include history supervision. A targeted Phase 4 with `freeze_except_history_bias=true` should train the asymmetric networks without disrupting the learned DDM parameters.
- Lapse (0.019 measured at 5% rollout) is improved from ~0.002 but below the 0.05 target; tuning the rollout lapse parameter to ~0.08 may help.

### Drift Calibration — Dead Knob Discovery (February 2026)

#### drift_calibration_v1: `drift_scale` has no effect

A sweep of `drift_scale`={12, 14, 16, 18} × 3 seeds (42, 123, 456) with the 3-phase curriculum produced:

| drift_scale | Psych Slope | Chrono Slope | Win-Stay | Lose-Shift |
|-------------|-------------|--------------|----------|------------|
| 12 | 21.75 ± 0.70 | -72.66 ± 4.02 | 0.556 | 0.572 |
| 14 | 22.42 ± 1.50 | -70.41 ± 1.93 | 0.557 | 0.563 |
| 16 | 21.30 ± 1.50 | -70.94 ± 3.27 | 0.558 | 0.556 |
| 18 | 21.24 ± 1.85 | -70.18 ± 4.43 | 0.556 | 0.557 |

**Psych slope is identical (~21.5) regardless of drift_scale.** This was unexpected — the parameter was assumed to control psychometric sensitivity.

**Root cause:** `drift_scale` only controls the *initialization* of the drift_head weights (lines 83-84 of `hybrid_model.py`). But the `drift_magnitude` loss in the trainer pulls `drift_gain` back to a **hardcoded target of 12.0** during training (`torch.mean((drift_gains - 12.0) ** 2)` with weight 0.5 in all three phases). The optimizer overrides the initialization every time. `drift_scale` is a dead knob.

#### drift_calibration_v2: `drift_magnitude_target` is the actual lever

The fix: make the regularization target configurable via `drift_magnitude_target` in `HybridTrainingConfig`. A sweep of `drift_magnitude_target`={6, 7, 8, 9} × 3 seeds:

| drift_magnitude_target | Psych Slope | Chrono Slope | Win-Stay | Lose-Shift |
|------------------------|-------------|--------------|----------|------------|
| **6.0** | **12.76 ± 1.04** | **-64.1 ± 2.4** | **0.556 ± 0.005** | **0.543 ± 0.004** |
| 7.0 | 15.19 ± 0.82 | -67.3 ± 3.2 | 0.557 ± 0.003 | 0.560 ± 0.021 |
| 8.0 | 15.96 ± 1.41 | -66.7 ± 3.6 | 0.563 ± 0.003 | 0.549 ± 0.009 |
| 9.0 | 17.08 ± 1.22 | -69.3 ± 2.6 | 0.557 ± 0.009 | 0.556 ± 0.005 |
| 12.0 (old default) | ~21.5 ± 1.5 | ~-70 ± 4.0 | ~0.556 | ~0.555 |
| **IBL target** | **13.2** | **negative** | **0.724** | **0.427** |

Clear monotonic relationship. **Target=6.0 gives psych slope 12.76 ± 1.04**, which brackets the IBL target of 13.2 (seed 123 hit 13.13).

#### Joint Production Test: Passed

The critical scientific question: does reducing drift sensitivity break the other fingerprints? **No.** At `drift_magnitude_target=6.0`:
- Chrono slope: -64.1 ± 2.4 ms/unit (unchanged from target=12)
- Win-stay: 0.556 (unchanged)
- Lose-shift: 0.543 (unchanged)
- Commit rate: 100%, bias: ~0.000

The architecture does not decouple when psychometric sensitivity is calibrated.

#### Scientific Framing

`drift_magnitude_target` is analogous to standard DDM fitting in neuroscience — drift rate is a property of sensory cortical hardware, not a learned strategy. The honest claim: we fit one parameter (evidence sensitivity) to match the animal's psychometric curve, and the remaining three behavioral fingerprints (chronometric slope, history effects, lapse) must emerge from architectural choices without additional fitting.

### Remaining Gaps

- **Psychometric slope**: ~~At 18.3 with drift=20 + 3-phase. Reducing drift to ~14-16 should reach ~13.2.~~ **Calibrated.** `drift_magnitude_target=6.0` → psych 12.76 ± 1.04.
- **Win-stay**: 0.556 vs target 0.724. Asymmetric architecture in place but untrained — needs history finetuning phase.
- **Lose-shift**: 0.543 vs target 0.427. Same — history finetuning needed.
- **Lapse**: ~0.025 measured at 5% rollout lapse. May need rollout parameter increase to ~0.08.

### Artifacts

- Original sweep directory: `runs/hybrid_sweep_ibl_drift_choice/` (9 configs, single-seed)
- Multi-seed validation: `runs/sweep_psych_slope_v1/` (5 seeds, drift=20/choice=0.5)
- Drift calibration sweep (old arch): `runs/sweep_drift_v2/` ({10,12,14} × 3 seeds)
- Learnable lapse sweep: `runs/sweep_lapse_v1/` ({25,30,35} × 3 seeds, learnable lapse — negative result)
- Single validation (new arch, drift=20): `runs/ibl_hybrid_curriculum/` (asymmetric history + learnable lapse)
- Fixed lapse sweep (training+rollout): `runs/sweep_fixed_lapse_v1/` ({20,22,25} × 3 seeds — training lapse double-counting)
- Rollout-only lapse sweep (7-phase): `runs/sweep_rollout_lapse_v1/` ({20,22,25} × 3 seeds — curriculum confound identified)
- **Curriculum control experiment**: `runs/sweep_3phase_newarch/` (drift=20 × 3 seeds — proved 7-phase was the problem)
- **Drift calibration v1 (dead knob)**: `runs/drift_calibration_v1/` (drift_scale={12,14,16,18} × 3 seeds — proved drift_scale is inert)
- **Drift calibration v2 (target sweep)**: `runs/drift_calibration_v2/` (drift_magnitude_target={6,7,8,9} × 3 seeds — calibrated psych slope)

---

```text
AnimalTaskSim: A Benchmark for Evaluating Behavioral Replication in AI Agents
https://github.com/ermanakar/animaltasksim
October 2025 – February 2026
Registry: 60+ experiments spanning Sticky-Q, PPO, Bayes Observer, Hybrid DDM+LSTM, and R-DDM agents
```
