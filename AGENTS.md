# Agent Operating Guide

## Mission Context

- AnimalTaskSim benchmarks AI agents against rodent/primate behavioral fingerprints using task-faithful environments, baseline agents, and a shared evaluation stack.
- MVP scope remains the IBL-style mouse 2AFC and macaque RDM tasks with Sticky-Q, Bayesian observer, and PPO baselines plus reproducible metrics/reports.
- Roadmap (v0.2) adds **Probabilistic Reversal Learning** and **Delayed Match-to-Sample**; design current work so these extensions slot in without breaking interfaces.
- Every contribution should strengthen the `.ndjson`-driven comparison pipeline between agents and animal data.

## Before You Touch Code

- Re-read `README.md`, `PRD.md`, `kickoff_prompt.md`, and task-specific specs; note v0.2 expectations even if they are out of scope now.
- Confirm changes stay inside the frozen CLI + schema contract; get explicit approval before renaming flags, paths, or log fields.
- List assumptions/upcoming questions early and surface blockers before implementation.

## Operating Principles

- Fidelity over flash: honor task timing, trial phases, priors, and response rules exactly as specified.
- Fingerprints over reward: optimise for psychometric/chronometric curves, history kernels, lapses/bias metrics, not raw reward.
- Reproducibility is mandatory: deterministic seeding, saved configs, schema-validated logging, CPU-friendly demos (<20 min, <4 GB RAM).
- Separation of concerns: keep envs, agents, metrics, scripts, and validators decoupled yet interoperable.
- Contracts are frozen: treat CLI arguments, file locations, and schema keys as immutable unless stakeholders approve a change.

## Standard Workflow

1. **Clarify & Plan** – Restate objectives, constraints, and success criteria; outline a multi-step plan unless the task is trivial.
2. **Design** – Sketch data structures, configs, logging hooks, and seeding strategy; make sure additions dovetail with schema validation and future PRL/DMS needs.
3. **Implement** – Write typed, docstring-rich Python 3.11 modules; prefer small, testable functions and keep comments purposeful.
4. **Validate** – Run targeted `pytest` suites (including `tests/test_schema.py`), smoke the relevant CLIs (`scripts/train_agent.py`, `scripts/evaluate_agent.py`, `scripts/make_report.py`), and confirm `.ndjson` logs pass `eval/schema_validator.py` while flushing each line immediately.
5. **Document & Handoff** – Update README/docs when behavior or interfaces change (or explicitly confirm no change), and report results, residual risks, and next steps.

## Implementation Standards

- Language: Python 3.11 with type hints and docstrings; keep notebooks out of core code.
- Dependencies: gymnasium, numpy, scipy, pandas, torch (CPU), stable-baselines3, matplotlib, pydantic, pytest, tyro/argparse.
- Configs: expose hyperparameters via dataclasses/pydantic; persist `config.json` next to run artifacts.
- Seeding: provide a single entrypoint that seeds Python, NumPy, torch, and envs; record the seed inside configs and logs.
- Logging: emit one JSON object per line, include all required schema keys (`task`, `session_id`, `trial_index`, `stimulus`, `block_prior`, `action`, `correct`, `reward`, `rt_ms`, `phase_times`, `prev`, `seed`, `agent{name,version}`), reject missing/extra keys, and `flush()` after each write.
- Validation: reuse `eval/schema_validator.py` in runtime checks and ensure `tests/test_schema.py` covers new log variants.
- Performance: default to CPU-only; ensure demo configs finish within the stated time/memory budgets.
- Interface stability: **never** rename CLI flags, change default paths, or alter schema types without written approval and migration guidance.

## Module Expectations

- **Environments (`envs/`):** Provide Gymnasium-compliant envs with correct observation/action spaces, phase timing, response windows, and logging hooks; share timing utilities via `envs/utils_timing.py`; anticipate reversible modes for PRL.
- **Agents (`agents/`):** Implement Sticky-Q (with stickiness term), Bayesian observer (sensory noise + lapses), and PPO baseline (action masking). Training loops must respect configs/seeds and write schema-compliant `.ndjson`.
- **Evaluation (`eval/`):** Maintain metrics for psychometric, chronometric, history, and bias analyses; extend fitters responsibly; keep `eval/schema_validator.py` authoritative for log validation; ensure `report.py` renders HTML with embedded metrics JSON.
- **Scripts (`scripts/`):** `train_agent.py`, `evaluate_agent.py`, and `make_report.py` must honour frozen CLIs, persist configs/logs/reports, and remain composable for future PRL/DMS runs.
- **Tests (`tests/`):** Cover env invariants, metric correctness on controlled data, agent smoke runs, and schema validation (`test_schema.py`); add fixtures when new schema fields arrive (e.g., PRL reversal flags, DMS delays).

## Deliverable Checklist

- Code formatted and typed, with purposeful comments and no stray TODOs.
- Config updates documented and stored alongside outputs.
- `.ndjson` logs validated via `eval/schema_validator.py`; schema test passes; logs flush cleanly without extra keys.
- Relevant tests and CLI demos executed; if sandbox restrictions block execution, state what could not run and why.
- README/PRD/kickoff alignment verified; docs updated when behavior changes.
- Roadmap readiness noted: call out how work supports upcoming PRL/DMS additions.

## Communication & Change Control

- Flag scope creep, schema/interface changes, or conflicting requirements immediately.
- Document any approved deviations from specs in commits/PR notes with mitigation plans.
- Keep handoff notes concise: what changed, why, validation performed, remaining risks, and how it aligns with future roadmap.

## Out of Scope Unless Explicitly Requested

- Web/hosted leaderboard or evaluation services.
- GPU or non-CPU acceleration paths.
- Neural data fitting or spike-based models.
- Implementing PRL/DMS deliverables ahead of schedule without approval.
- Altering CLI surfaces, schema keys, or log formats.
