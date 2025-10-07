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
