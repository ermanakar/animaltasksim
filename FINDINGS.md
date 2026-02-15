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

This is a breakthrough relative to all prior runs:

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

```text
AnimalTaskSim: A Benchmark for Evaluating Behavioral Replication in AI Agents
https://github.com/ermanakar/animaltasksim
October 2025 – February 2026
Registry: 55+ experiments spanning Sticky-Q, PPO, Bayes Observer, Hybrid DDM+LSTM, and R-DDM agents
```
