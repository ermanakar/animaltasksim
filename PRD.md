# PRD — AnimalTaskSim v0.1 (MVP)

## Problem
AI agents routinely hit high reward rates on toy tasks, but fail to reproduce **animal behavioral fingerprints** (history effects, lapses, priors, speed–accuracy tradeoffs). There’s no unified, reproducible benchmark that (a) mirrors real rodent/primate paradigms, (b) trains agents under those constraints, and (c) **scores them** with the metrics neuroscientists actually use.

## Goal (MVP)
Ship two task-faithful environments (**IBL-style mouse 2AFC**, **macaque RDM**), three baseline agents, and an evaluation suite that outputs a **replication report** comparing agents to animal-style statistics.

## Non-Goals (MVP)
- Not a physics-accurate 3D simulator.
- Not a neural data model (no spike fitting).
- No web leaderboard yet (local HTML reports only).

## Users
- **Neuroscience method developers:** want a yardstick to test models of behavior.  
- **RL researchers/engineers:** want to claim “animal-like” with evidence.  
- **Educators:** need course-ready labs.

## User Stories
- As an RL engineer, I can train `ppo` on `ibl_2afc` and get an HTML report with psychometric + history metrics.  
- As a neuroscientist, I can run a **Bayesian observer** baseline and see lapse/bias parameters comparable to literature ranges.  
- As an instructor, I can run a 10-minute demo and export a PDF report.

## Scope (v0.1)
- `envs/ibl_2afc.py`, `envs/rdm_macaque.py`
- `agents/sticky_q.py`, `agents/bayes_observer.py`, `agents/ppo_baseline.py`
- `eval/metrics.py`, `eval/report.py`, `eval/fitters.py`
- CLI scripts, tests, Quickstart notebook

## Success Metrics
- **S1:** Psychometric curves render and fit without errors on both tasks.  
- **S2:** History kernels for `sticky_q` show non-zero perseveration; PPO shows near-zero lapses without noise injection.  
- **S3:** RDM chronometric curve exhibits speed–accuracy tradeoff (with urgency/cost).  
- **S4:** Full demo (train+eval) runs < 20 minutes CPU-only, < 4GB RAM.

## Constraints
- Python 3.11; Gymnasium; Stable-Baselines3; torch CPU-only by default.
- Deterministic seeds & config export.
- `.ndjson` trial log schema is mandatory.

## Risks & Mitigations
- **R1: Superficial match (accuracy only).** Metrics prioritize fingerprints; reports highlight mismatches.  
- **R2: Data ambiguity for primates.** Use canonical paradigm stats for ranges; label assumptions; keep parameters configurable.  
- **R3: Scope creep.** Two tasks only; week-2 and week-4 gates.

## Timeline
- Wk 1: IBL env + schema + Sticky-Q + metrics skeleton.  
- Wk 2: Polishing + first replication report.  
- Wk 3: RDM env + PPO baseline.  
- Wk 4: Bayesian observer + chronometric + second report.

## Deliverables
- Repo with envs/agents/eval + tests.  
- Two HTML reports (one per task).  
- Quickstart notebook.  
- Docs (README).

## Open Questions
- Should we expose block priors to agents during ablations by default? (default: no)  
- Which K for history kernels? (default: 5)  
- Do we include a minimal web viewer for reports in v0.1? (default: no)

## v0.2 Scope (post‑MVP) — PRL & DMS

Extend the benchmark with two cognitive axes that reuse the MVP scaffold: **flexibility** in mice via **Probabilistic Reversal Learning (PRL)** and **working memory** in macaques via **Delayed Match‑to‑Sample (DMS)**.

### 1) Mouse — Probabilistic Reversal Learning (PRL)
**Goal:** Test cognitive flexibility; quantify perseveration vs regression when reward contingencies flip.

**Implementation**
- Option A: add `--reversal` mode to `envs/ibl_2afc.py`; Option B: create `envs/reversal_2afc.py` reusing 2AFC utilities.
- Two actions (left/right). Reward probabilities per block: `p_hi=0.8`, `p_lo=0.2` (configurable).
- Reverse after fixed `block_len` **or** after a performance `criterion` (e.g., ≥8/10 correct), with optional jitter.
- **Logging additions:** `block_id:int`, `is_reversal_trial:bool`, `reversal_index:int|null` in the `.ndjson` records.

**Metrics (to add in `eval/metrics.py`)**
- **Trials‑to‑criterion** post‑reversal (median & distribution).
- **Perseverative errors** (errors while following the old rule immediately post‑reversal).
- **Regressive errors** (returns to old rule after initial adoption of the new rule).
- **Win‑stay / Lose‑shift** rates per block; asymmetry index.
- **Reward‑sensitivity vs choice‑stickiness** (logistic with previous reward & previous action predictors).

**Baselines**
- `sticky_q` (expected: strong perseveration after reversal).
- Belief‑tracking / HMM reversal detector (reduces perseveration).

**Acceptance criteria**
- Demo (CPU) ≤ 10 min default config.
- Report contains trials‑to‑criterion and error‑type panels; logs include the new fields.
- Sticky‑Q shows a perseverative burst; belief‑tracking agent measurably reduces it.

### 2) Macaque — Delayed Match‑to‑Sample (DMS)
**Goal:** Test working memory with a variable **delay** between sample and test.

**Implementation**
- New env `envs/dms_macaque.py`.
- Phases: `fixation → sample(S) → delay(Δ) → test(T) → response → outcome`.
- Actions: `{match, nonmatch, hold}`; enforce **hold** during delay (action masking in PPO wrapper).
- Config: `delays=[250,500,1000,2000,3000]` ms, `lure_similarity∈{easy,hard}`, optional go‑cue.
- Observations: expose IDs/features during sample & test; mask/minimize during delay; always log `rt_ms`.

**Metrics**
- **Accuracy vs delay** (forgetting curve) + fitted slope/half‑life.
- **RT vs delay** (chronometric slope/intercept).
- **Lure‑specific errors** (confusion matrix by similarity).
- **Interference**: error rate as a function of recent‑trial similarity.

**Baselines**
- Memoryless observer (upper‑chance only when trivial cues exist).
- PPO‑LSTM / recurrent policy expected to outperform feed‑forward PPO at longer delays.

**Acceptance criteria**
- Demo (CPU) ≤ 15 min default config.
- Feed‑forward PPO accuracy declines with longer delays; PPO‑LSTM shows a smoother decline.
- Report includes delay‑accuracy and RT curves; logs are schema‑compliant with `rt_ms`.

### Deliverables (v0.2)
- `envs/reversal_2afc.py` (or `ibl_2afc` with reversal mode) and `envs/dms_macaque.py`.
- Metric & report extensions; unit tests for new metrics and schema fields.
- Example HTML reports for both tasks.

### Timeline (indicative)
- **Wk 5–6:** PRL env + metrics + sticky‑Q & HMM baselines → report.
- **Wk 7–8:** DMS env + PPO/PPO‑LSTM baselines → report.

### Risks & Mitigations
- **Overfitting reversal schedule:** randomize block lengths; validate on held‑out schedules.
- **Working‑memory shortcuts:** mask/normalize non‑diagnostic cues; add lure‑similarity controls; verify no leakage via logs.
