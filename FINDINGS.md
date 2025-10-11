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

## Macaque RDM — Hybrid DDM+LSTM Reference Point

- **Run**: `runs/rdm_wfpt_regularized/`
- **Objective**: Demonstrate that adding a mechanistic accumulator plus WFPT likelihood yields coherence-dependent RTs, albeit with calibration errors.

| Metric | Hybrid Agent | Reference | Gap |
| --- | --- | --- | --- |
| RT intercept | 1.26 s | 0.76 s | 500 ms slower |
| RT slope | −981 ms/unit | −645 ms/unit | Overshoots desired slope |
| Psychometric slope | 10.9 | 17.6 | Too shallow |
| Bias | −0.001 | ≈0 | Matches |

### Interpretation (Hybrid)

- Evidence-dependent timing emerges once the agent learns to integrate noisy drift, but calibration is off. The intercept overshoot and shallow choice slope point to non-decision-time and drift-scale mismatches.
- The run remains the closest match to animal chronometric structure, underscoring the need for explicit accumulation rather than reactive flight to action.

Artifacts: `runs/rdm_wfpt_regularized/metrics.json`, `dashboard.html`.

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

## Next Steps

1. **Agent-side latency models**: Give Sticky-Q and PPO explicit non-decision-time estimates that can adapt by coherence or block, rather than relying on environment delays.
2. **Bias and lapse regularizers**: Penalize large bias deviations and encourage lapse rates that match observed animal floors instead of zero or fifty percent extremes.
3. **History-aware losses**: Incorporate win-stay/lose-shift targets or kernel regression loss terms so agents learn multi-trial dependencies present in the data.
4. **Roadmap readiness**: Generalize logging and evaluation hooks so Probabilistic Reversal Learning and Delayed Match-to-Sample tasks can reuse the same schema without touching CLI surfaces.

---

## Artifact Index

- Mouse Sticky-Q latency run: `runs/ibl_stickyq_latency/`
- Macaque PPO baseline rerun: `runs/rdm_ppo_latest/`
- Hybrid DDM+LSTM comparison: `runs/rdm_wfpt_regularized/`
- Reference data: `data/ibl/reference.ndjson`, `data/macaque/reference.ndjson`
- Evaluation utilities: `scripts/evaluate_agent.py`, `scripts/make_report.py`, `scripts/make_dashboard.py`

All directories contain `config.json`, `trials.ndjson`, `metrics.json`, and at least one HTML artifact for inspection.

---

## Citation

```text
AnimalTaskSim: A Benchmark for Evaluating Behavioral Replication in AI Agents
https://github.com/ermanakar/animaltasksim
October 2025
```
