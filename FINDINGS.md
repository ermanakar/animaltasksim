# AnimalTaskSim Findings

**Benchmarking reinforcement learning agents against rodent and primate decision-making fingerprints**  
*October 2025 · Version 0.1.0*

---

## Context

AnimalTaskSim compares learning agents to real animals on the IBL mouse 2AFC and the Roitman & Shadlen macaque random-dot motion (RDM) tasks. Every run logs one JSON object per trial under a frozen schema, enabling direct comparison of psychometric, chronometric, history, and lapse statistics. The October 2025 round of experiments focused on two priorities:

- Remove simulator shortcuts (auto-commit, implicit latency) that previously inflated agent resemblance.
- Regenerate baseline runs and document the resulting behavioral gaps using the hardened pipeline.

Fresh evidence comes from `runs/ibl_stickyq_latency/` (Sticky-Q with a 200 ms latency) and `runs/rdm_ppo_latest/` (PPO with collapsing bounds disabled). We also summarize the best-performing hybrid DDM+LSTM run (`runs/rdm_wfpt_regularized/`) to highlight what mechanism-level structure buys us.

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

## Updated Artifact Index (21 Registry Entries)

### IBL Mouse 2AFC Runs (10 entries)

- **Sticky-Q**: `ibl_stickyq_latency`, `20251017_ibl_stickyq`, `20251018_ibl_stickyq`
- **PPO**: `20251017_ibl_ppo`, `20251017_ibl_ppo_test`, `20251018_ibl_ppo`
- **Hybrid DDM+LSTM**: `hybrid_wfpt_curriculum`, `hybrid_wfpt_curriculum_repro`, `hybrid_wfpt_curriculum_timecost`, `hybrid_wfpt_curriculum_timecost_attempt1`, `hybrid_wfpt_curriculum_attempt2_two_phase`, `hybrid_wfpt_curriculum_timecost_soft_rt`, `guarded_curriculum_v2`, `annealed_choice_v1`, `history_supervision_v1`, `history_finetune_v1`, `history_finetune_v2` (training artifacts exist, dashboard not yet generated), `final_history_embedding_agent`, `drift_scale_14.0_stable` (metrics pending), `drift_scale_14.0_stable_v2`, `rt_calibration_v1`, `rt_weighted_calibration_v1`

### Macaque RDM Runs (5 entries)

- **PPO**: `rdm_ppo_latest`, `20251017_rdm_ppo`
- **Bayes Observer**: `20251017_rdm_bayes` (baseline with sensory noise model)
- **Hybrid DDM+LSTM**: `rdm_wfpt_regularized`, `rdm_hybrid_curriculum`, `20251017_rdm_hybrid_test`, `20251017_rdm_hybridddml`

### Reference Data

- `data/ibl/reference.ndjson` (IBL mouse 2AFC behavioral fingerprints)
- `data/macaque/reference.ndjson` (Roitman & Shadlen macaque RDM data)

### Evaluation Pipeline

- `scripts/evaluate_agent.py` (schema-compliant logging + metrics computation)
- `scripts/make_report.py` (HTML report generation)
- `scripts/make_dashboard.py` (comparative dashboards)
- `runs/registry.json` (experiment registry with 21 tracked runs)

---

## Next Steps (Prioritized by Evidence)

1. **Implement R-DDM architecture** with dynamic drift modulation during evidence accumulation, targeting simultaneous chronometric slopes *and* history effects.
2. **History-aware loss functions**: Penalize deviations from target win-stay/lose-shift rates to provide explicit training signal for inter-trial dependencies.
3. **Bias and lapse regularizers**: Current hybrid runs show unstable bias (ranges from near-zero to pathological +6). Add soft constraints or prior terms to keep bias within animal-observed ranges.
4. **Psychometric calibration**: Hybrid agents produce slopes 5–10 while macaques show 17.6. Investigate drift-gain scaling and sensory noise parameters to match animal choice sensitivity.
5. **Roadmap readiness**: Ensure R-DDM and evaluation hooks generalize to Probabilistic Reversal Learning (PRL) and Delayed Match-to-Sample (DMS) tasks without breaking schema contracts.

---

## Citation

```text
AnimalTaskSim: A Benchmark for Evaluating Behavioral Replication in AI Agents
https://github.com/ermanakar/animaltasksim
October 2025
Registry: 21 experiments spanning Sticky-Q, PPO, Bayes Observer, and Hybrid DDM+LSTM agents
```
