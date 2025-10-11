# AnimalTaskSim

AnimalTaskSim is a reproducible benchmark for asking a simple question: **do our learning agents behave like the animals that solved the original laboratory tasks?** We provide task-faithful environments, public reference datasets, seeded baselines, and analysis tools that turn raw trial logs into behavioral fingerprints (psychometric, chronometric, history, and lapse statistics).

---

## Project Goals

- Mirror published mouse and macaque decision-making tasks with Gymnasium environments that respect timing, response rules, and priors.
- Train baseline agents under fixed seeds, logging one schema-validated JSON object per trial.
- Compare agent logs against the animal reference data with the same metrics neuroscientists use.
- Surface where agents succeed or fail so model improvements can target real behavioral gaps.

---

## October 2025 Status Snapshot

We reran both benchmarks after hardening the environments (collapsing bounds are now disabled by default and the mouse task supports an explicit non-decision latency). Fresh dashboards live in `runs/ibl_stickyq_latency/` and `runs/rdm_ppo_latest/`.

| Task & Agent | Key Agent Metrics | Reference Metrics | Takeaway |
| --- | --- | --- | --- |
| Mouse 2AFC — Sticky-Q with 200 ms latency | Bias −0.0001; psychometric slope 33.3; RT median 210 ms flat across contrasts; win-stay 0.67 | Bias +0.074; slope 13.2; RT median 300 ms with decreasing slope −36 ms/unit; win-stay 0.73 | Latency floor produces realistic RT scale, but the agent still commits immediately once allowed, overshoots contrast sensitivity, and over-perseverates after wins. |
| Macaque RDM — PPO (collapsing bound disabled) | Bias +0.52; psychometric slope 50.0; RT median 60 ms at all coherences; lapse_low 0.49 | Bias ≈0; slope 17.6; RT median 760 ms with slope −645 ms/unit; lapse_low ≈0 | Removing the auto-commit reveals that PPO still collapses to minimum response time and relies on large lapses instead of evidence accumulation. |
| Macaque RDM — Hybrid DDM+LSTM (`runs/rdm_wfpt_regularized/`, Oct 2025) | RT slope −981 ms/unit; intercept 1.26 s; psychometric slope 10.9 | RT slope −645 ms/unit; intercept 0.76 s; slope 17.6 | Mechanistic accumulation yields coherence-dependent timing, but absolute RTs remain too slow and choice slope too shallow. |

Bottom line: the infrastructure now reflects genuine agent behavior, but no current baseline matches animal timing and bias simultaneously.

---

## Quickstart

```bash
# Install (editable + dev extras)


# Mouse baseline with a 200 ms non-decision latency
python scripts/train_agent.py \
  --env ibl_2afc --agent sticky_q \
  --steps 8000 --trials-per-episode 400 \
  --sticky-min-response-latency-steps 20 \
  --out runs/ibl_stickyq_latency

# Macaque PPO baseline (collapsing bound disabled by default)
python scripts/train_agent.py \
  --env rdm --agent ppo \
  --steps 200000 --trials-per-episode 300 \
  --out runs/rdm_ppo_latest

# Compute metrics and render artifacts
python scripts/evaluate_agent.py --run runs/ibl_stickyq_latency
python scripts/make_dashboard.py \
  --opts.agent-log runs/ibl_stickyq_latency/trials.ndjson \
  --opts.reference-log data/ibl/reference.ndjson \
  --opts.output runs/ibl_stickyq_latency/dashboard.html
```

All scripts seed Python, NumPy, and PyTorch, persist `config.json`, and emit `.ndjson` logs that pass `eval/schema_validator.py`.

---

## Repository Layout

```text
animaltasksim/
├─ envs/                # Gymnasium tasks + timing utilities
├─ agents/              # Sticky-Q, Bayesian observer, PPO, hybrid DDM
├─ eval/                # Metrics, dashboards, schema validator
├─ scripts/             # Train / evaluate / make_report / make_dashboard
├─ data/                # Reference animal logs (IBL mouse, Roitman macaque)
├─ runs/                # Generated configs, logs, metrics, dashboards
└─ tests/               # Env/metric/schema unit tests
```

---

## Task Overview

### IBL Mouse 2AFC

- Actions: `left`, `right`, `no-op`; observations include signed contrast (−1 to 1) with optional phase and timing encodings.
- `min_response_latency_steps` (default 0) can be set to impose a non-decision delay before actions register.
- Block priors and contrast ladder replicate the public IBL dataset (`data/ibl/reference.ndjson`).

### Macaque Random-Dot Motion

- Actions: `left`, `right`, `hold` with streaming evidence during stimulus and response phases.
- Collapsing bounds now default to `False`; agents must decide when to commit instead of inheriting the simulator’s accumulator.
- Optional reward shaping (confidence bonuses, RT targets) is captured in saved configs for reproducibility.

---

## Evaluation Workflow

1. **Train** with `scripts/train_agent.py` (or task-specific scripts). Each run lands in `runs/<name>/` with `config.json`, `trials.ndjson`, and training metrics.
2. **Score** with `scripts/evaluate_agent.py` to produce `metrics.json` (psychometric, chronometric, history summaries; NaNs are exported as `null`).
3. **Inspect** with `scripts/make_report.py` (static HTML) or `scripts/make_dashboard.py` (interactive comparison against animal data).
4. **Validate** any log with `eval/schema_validator.py` or `pytest tests/test_schema.py`.

The benchmark emphasises visible trial-by-trial evidence: every dashboard links back to the exact `.ndjson` and config that generated it.

---

## Scientific Caveats

- Agents still exploit shortcuts. Sticky-Q commits on the first eligible step, so the latency parameter only shifts the intercept; slopes remain flat.
- PPO requires additional structure (e.g., explicit accumulators) to express animal-like RT slopes; reward shaping alone is insufficient.
- Hybrid DDM runs show the desired trend but highlight calibration gaps (non-decision time, low-coherence handling).
- Metrics depend on reliable fits; we treat failed logistic regressions as `null` to keep downstream analyses honest.

---

## Roadmap

- Extend the evaluation stack to Probabilistic Reversal Learning and Delayed Match-to-Sample tasks (interfaces already anticipate these additions).
- Tighten RT modelling by adding non-decision components to agents rather than the environment.
- Provide parameter sweeps and ablation scripts so others can reproduce the negative results as readily as the positive ones.

---

## License

Code is released under the MIT License. Reference datasets retain their original licenses and attribution requirements.
