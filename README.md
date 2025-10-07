# AnimalTaskSim

AnimalTaskSim is a **benchmark** for reproducing classic animal behavioral tasks in silico and scoring AI agents against **animal-style fingerprints** — not just reward rate. It ships **task-faithful environments**, **baseline agents**, and an **evaluation suite** that compares agents to mice/primate benchmarks on psychometric/chronometric curves, history effects, and training dynamics.

> Scope for v0.1 (MVP): two tasks
>
> 1) **Mouse 2AFC visual decision (IBL-style)**  
> 2) **Macaque random-dot motion discrimination (RDM)**

We aim to make “replication with teeth” the standard: codify the task, train the agent under the same constraints, then grade it with the *same* metrics labs use.

---

## Why this exists

- Toolkits like NeuroGym/PsychRNN and virtual labs like Animal-AI are valuable, but none combine **task-faithful environments + public animal data + fingerprint metrics + a reproducible scoring pipeline**.
- Labs and RL teams need a **defensible yardstick** for “animal-like behavior.” Reward rate alone is vanity; fingerprints are substance.

---

## Repo structure (planned)

```
animal-task-sim/
├─ envs/
│  ├─ ibl_2afc.py            # Gymnasium env for mouse 2AFC
│  ├─ rdm_macaque.py         # Gymnasium env for RDM
│  └─ utils_timing.py        # Common trial/timing utilities
├─ agents/
│  ├─ sticky_q.py            # Q-learning + stickiness baseline
│  ├─ bayes_observer.py      # Ideal observer with sensory noise
│  └─ ppo_baseline.py        # Stable-Baselines3 PPO wrapper
├─ eval/
│  ├─ metrics.py             # Psychometric/chronometric, history kernels
│  ├─ fitters.py             # Logistic fits, RT fits, kernel regressions
│  ├─ report.py              # End-to-end replication report generation
│  └─ simlog_schema.json     # Unified trial-log schema
├─ data/
│  ├─ ibl/
│  │  └─ download_ibl.py     # Script to fetch example IBL behavior slices
│  └─ macaque/
│     └─ references.md       # Protocol references & expected ranges
├─ scripts/
│  ├─ train_agent.py         # Train any agent on any env via CLI
│  ├─ evaluate_agent.py      # Run eval suite and save report
│  └─ make_report.py         # Pretty HTML/PDF report
├─ notebooks/
│  ├─ 00_quickstart.ipynb
│  └─ 10_recreate_figures.ipynb
├─ tests/
│  ├─ test_envs.py
│  ├─ test_metrics.py
│  └─ test_agents.py
├─ pyproject.toml
├─ README.md
└─ LICENSE
```

---

## Task specs (v0.1)

### 1) Mouse visual 2AFC (IBL-style)

**Goal:** choose left vs right based on signed contrast. Sessions mix **block priors** (e.g., 80/20 left/right) and **neutral** blocks.  
**Trial phases:** ITI → fixation (optional) → stimulus (contrast) → response window → outcome (reward/timeout).

**Observation (default):**
- `contrast`: float in [-1,1] (sign = side; magnitude = difficulty)
- `phase`: one-hot of trial phase (optional)
- `t_in_phase`: normalized time within phase (optional)

> Note: block prior is **not** exposed to the agent by default (mirrors animals). A diagnostic flag `expose_prior` can be toggled for ablations.

**Action space:** `Discrete(3)` → {`left`, `right`, `no-op`}  
**Reward:** `+1` on correct choice; `0` otherwise; configurable timeouts / ITI penalties.  
**Episode:** fixed-length session (e.g., 400 trials) → `done=True`.

**Key parameters:**
- Contrast set: e.g., `{±0, 6.25, 12.5, 25, 50, 100}%` mapped to [-1,1]
- Block structure: sequences of biased vs neutral blocks
- ITI, response window, timeouts (all in steps or seconds → discretized)

### 2) Macaque random-dot motion (RDM)

**Goal:** decide left vs right based on motion direction at coherence `c`.  
**Trial phases:** fixation → stimulus (dots) → (optional) go cue → response → outcome.

**Observation (default):**
- `coherence`: float in [0,1] (signed by direction)
- `phase` and `t_in_phase` as above (optional)

**Action space:** `Discrete(3)` → {`left`, `right`, `hold`}  
**Reward:** `+1` on correct; optional small cost per time-step to incentivize speed–accuracy tradeoff.  
**Timing:** max stimulus duration; optional collapsing decision bound implemented as reward shaping or termination rule.

---

## Unified trial-log schema

Every run writes newline-delimited JSON (`.ndjson`) with the following fields per trial:

```json
{
  "task": "ibl_2afc|rdm",
  "session_id": "uuid",
  "trial_index": 42,
  "stimulus": {"contrast": 0.25, "side": "right"},
  "block_prior": {"p_right": 0.8},
  "action": "left",
  "correct": false,
  "reward": 0.0,
  "rt_ms": 540,
  "phase_times": {"stim_ms": 300, "resp_ms": 700},
  "prev": {"action": "right", "reward": 1},
  "seed": 1234,
  "agent": {"name": "sticky_q", "version": "0.1.0"}
}
```

This fuels the evaluation suite independent of the agent/env internals.

---

## Evaluation metrics (v0.1)

- **Psychometric curve**: P(choice=right) vs signed contrast; fit logistic → report **threshold**, **slope**, **lapse**.
- **Chronometric curve** (RDM): median RT vs coherence; report **speed–accuracy tradeoff** stats.
- **History fingerprints**: win-stay/lose-shift rates; **sticky-choice**; logistic **history kernel regression** (past K choices/outcomes).
- **Bias & priors**: block-prior effect size on choice bias.
- **Training dynamics**: trials-to-criterion; early vs late session performance deltas.

Output: JSON metrics + an HTML report that recreates two canonical figures per task (psychometric + history; plus chronometric for RDM).

---

## Baseline agents (v0.1)

1) **Sticky-Q**: tabular Q-learning with a **stickiness term** in the choice rule.  
2) **Bayesian observer**: ideal observer with sensory noise; softmax decision with lapses.  
3) **PPO baseline**: Stable-Baselines3 PPO with masked actions during non-response phases.

Each baseline must:
- Save seeds & configs
- Log to `.ndjson` schema
- Train ≤ 20 minutes on CPU for a demo run (small configs)
- Produce a valid evaluation report via `scripts/evaluate_agent.py`

---

## Installation (dev)

```bash
pyenv local 3.11.9         # or your preferred Python 3.11
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]       # via pyproject optional extras
```

**Key deps:** `gymnasium`, `numpy`, `scipy`, `pandas`, `torch`, `stable-baselines3`, `matplotlib`, `tyro` (or `argparse`), `pydantic` for config, `pytest`.

---

## CLI

```bash
# Install
pip install -e ".[dev]"

# Train Sticky-Q on IBL 2AFC (writes runs/ibl_stickyq)
python scripts/train_agent.py --env ibl_2afc --agent sticky_q --steps 2000 --out runs/ibl_stickyq

# Train Bayesian observer on RDM
python scripts/train_agent.py --env rdm --agent bayes --steps 1200 --out runs/rdm_bayes

# Train PPO baseline on RDM with a per-step cost
python scripts/train_agent.py --env rdm --agent ppo --steps 10000 --trials-per-episode 200 --ppo.per-step-cost 0.05 --out runs/rdm_ppo

# Evaluate (writes metrics.json)
python scripts/evaluate_agent.py --run runs/ibl_stickyq

# Generate HTML report
python scripts/make_report.py --run runs/ibl_stickyq

# Optional overrides
python scripts/evaluate_agent.py --log some_agent.ndjson --out reports/some_agent_metrics.json
python scripts/make_report.py --run runs/ibl_stickyq --log runs/ibl_stickyq/trials.ndjson --metrics runs/ibl_stickyq/metrics.json --out reports/custom.html
```

## Running baselines & comparing to reference data

1. Train an agent with `scripts/train_agent.py` to populate `runs/<name>/` with `trials.ndjson` and configs.
2. Evaluate that log (`scripts/evaluate_agent.py --run <run_dir>`) to produce `metrics.json`.
3. Generate an HTML report (`scripts/make_report.py --run <run_dir>`) to visualise psychometric/chronometric/history fingerprints alongside the metrics JSON.
4. Run the same evaluation on reference `.ndjson` logs under `data/ibl/` or `data/macaque/`, then compare the resulting metrics JSON to your agent (e.g. via a notebook or a short Python diff script).
5. Iterate on hyperparameters (`--sticky_q.stickiness`, `--bayes.sensory_sigma`, `--ppo.per-step-cost`, etc.) until the fingerprints align with target animal ranges.

---

## Milestones (time-boxed)

- **Week 1–2:** envs + Sticky-Q + metrics (psychometric, history) → first replication report for IBL-style task. **Gate:** curves look sane.  
- **Week 3–4:** PPO + Bayesian observer + RDM env + chronometric metrics → second report. **Gate:** PPO nails reward rate; mismatches at least one fingerprint; Bayesian moves it closer.

---

## Principles

- **Fidelity over flash:** copy trial structure and timing; keep visuals minimal.
- **Fingerprints over reward:** we celebrate matched **statistics**, not points scored.
- **Reproducibility:** fixed seeds, saved configs, exact software versions.
- **Separation of concerns:** envs, agents, and metrics do not depend on each other.

---

## Roadmap

**v0.1 (MVP)** — IBL 2AFC + Macaque RDM; baselines (Sticky‑Q, Bayesian observer, PPO); evaluation suite & HTML reports.

**v0.2 (next)** — cognitive axes that reuse the same scaffold:
- **Mouse PRL (flexibility):** reversal blocks with p=0.8/0.2; metrics include trials‑to‑criterion, perseverative/regressive errors, WS/LS asymmetry.
- **Macaque DMS (working memory):** sample→delay→test; metrics include accuracy vs delay, RT vs delay, lure‑specific errors.

> CLI (planned) examples:
> ```bash
> # PRL (as a mode of ibl_2afc)
> python scripts/train_agent.py --env ibl_2afc --agent sticky_q --steps 15000 --out runs/prl_stickyq --reversal true
> 
> # DMS
> python scripts/train_agent.py --env dms --agent ppo --steps 30000 --out runs/dms_ppo --delays 250,500,1000,2000,3000
> ```

---

## Ethics note

This does **not** claim to replace animal research. It’s a comparative benchmark that pressures AI to behave under animal-like constraints and makes replication transparent and reusable.

---

## License

MIT for code; dataset licenses remain with their owners.
