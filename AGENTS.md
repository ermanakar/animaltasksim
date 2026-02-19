# Agent Operating Guide

## Mission Context

- AnimalTaskSim benchmarks AI agents against rodent/primate behavioral fingerprints using task-faithful environments, baseline agents, and a shared evaluation stack.
- Current scope (v0.2): IBL mouse 2AFC and macaque RDM tasks with Sticky-Q, Bayesian observer, PPO baselines, Hybrid DDM+LSTM, and R-DDM agents. 93 tests, 60+ registered experiments.
- Roadmap (v0.2) adds **Probabilistic Reversal Learning** and **Delayed Match-to-Sample**; schema extensions are already drafted and tested (`tests/test_schema_v02.py`). Design so these slot in without breaking interfaces.
- Every contribution should strengthen the `.ndjson`-driven comparison pipeline between agents and animal data.

## Build & Test

```bash
pip install -e ".[dev]"        # install with dev deps (pytest, pytest-cov, ruff)
pytest                         # runs tests/ with -ra (configured in pyproject.toml)
pytest tests/test_schema.py    # schema validation specifically
```

Smoke-test the frozen CLIs after any change:

```bash
python scripts/train_agent.py --help
python scripts/evaluate_agent.py --run runs/<run_dir>
python scripts/make_report.py --run runs/<run_dir>
```

## Architecture

```text
envs/           → Gymnasium envs (IBL 2AFC, RDM). Env owns the NDJSONTrialLogger.
agents/         → Agent implementations (Sticky-Q, Bayes, PPO, DDM, Hybrid)
eval/           → Metrics, schema validation (authoritative), HTML reports
scripts/        → Frozen CLI entrypoints (train, evaluate, report)
animaltasksim/  → Shared core (config, logging, seeding, registry)
```

**Data flow:** Agent drives Env → Env logs trial via `NDJSONTrialLogger` (validate → write → flush) → `trials.ndjson` → `evaluate_agent.py` computes `metrics.json` → `make_report.py` renders HTML.

Key rule: the **environment** owns logging — agents never write `.ndjson` directly.

## Code Style

- **Python ≥ 3.11**, `from __future__ import annotations` at top of every file
- **Type hints everywhere**: PEP 604 unions (`str | None`), built-in generics (`dict[str, object]`)
- **`@dataclass(slots=True)`** for all config/data classes — never bare `@dataclass`
- **Docstrings**: single-line `"""..."""` module-level; Google-style Args/Returns when documenting params
- **Naming**: `UPPER_SNAKE` constants, `_underscore` private functions, `PascalCase` classes
- **`__all__`** explicitly declared in most modules
- **ruff** is the linter (dev dependency, default config)
- Exemplar files: `envs/ibl_2afc.py`, `agents/sticky_q.py`, `animaltasksim/logging.py`

## Project Conventions

### Frozen Contracts

CLI args, log file paths, and schema keys are **immutable**. Never rename flags, change default paths, or alter schema types without explicit approval.

### Schema (`eval/schema_validator.py`)

`TrialRecord` uses Pydantic `BaseModel` with `extra="forbid"` — no unexpected keys allowed. Required fields: `task`, `session_id`, `trial_index`, `stimulus`, `block_prior`, `action`, `correct`, `reward`, `rt_ms`, `phase_times`, `prev`, `seed`, `agent{name, version}`.

### Logging (`animaltasksim/logging.py`)

`NDJSONTrialLogger`: validates each record via `TrialRecord.model_validate()`, writes compact JSON, calls `flush()` after every line. Used as context manager.

### Seeding (`animaltasksim/seeding.py`)

`seed_everything(seed)` seeds `random`, `numpy`, `torch`, and sets `PYTHONHASHSEED`. Called once at agent training start. Per-episode resets use `seed=config.seed + episode`.

### Config (`animaltasksim/config.py`)

`ProjectPaths.from_cwd()` provides standard directory layout (`root`, `data`, `runs`, `logs`). Agent/env configs define `output_paths() -> dict[str, Path]` that creates directories and returns path dict. Configs are serialized to `config.json` alongside run artifacts.

### CLIs (`scripts/`)

All use `tyro.cli(DataclassArgs)` — no manual argparse. `Literal` types restrict valid choices. Nested dataclass fields become CLI sub-groups (e.g., `--sticky-q.learning-rate`).

### Testing (`tests/`)

- Tests use `tmp_path` fixture for isolated I/O
- Every agent test validates output with `validate_file(log_path)` — schema compliance is a first-class assertion
- Agent tests also run the full evaluate pipeline to confirm end-to-end
- Test helpers prefixed with `_` (e.g., `_write_ndjson`, `_make_psychometric_df`)

## Operating Principles

- Fidelity over flash: honor task timing, trial phases, priors, and response rules exactly as specified.
- Fingerprints over reward: optimise for psychometric/chronometric curves, history kernels, lapses/bias metrics, not raw reward.
- Reproducibility is mandatory: deterministic seeding, saved configs, schema-validated logging, CPU-friendly demos (<20 min, <4 GB RAM).
- Separation of concerns: keep envs, agents, metrics, scripts, and validators decoupled yet interoperable.
- Contracts are frozen: treat CLI arguments, file locations, and schema keys as immutable unless stakeholders approve a change.

## Workflow

1. **Clarify & Plan** – Restate objectives and constraints; confirm changes stay inside frozen CLI + schema contract.
2. **Design** – Sketch data structures, configs, logging hooks; ensure additions dovetail with schema validation and future PRL/DMS needs.
3. **Implement** – Write typed Python 3.11 with `@dataclass(slots=True)` configs, `__all__` exports, and purposeful comments.
4. **Validate** – Run `pytest` (especially `test_schema.py`), smoke relevant CLIs, confirm `.ndjson` logs pass `eval/schema_validator.py`.
5. **Document** – Update README/docs when behavior changes; report results, risks, and roadmap alignment.

## Deliverable Checklist

- Code formatted, typed, `slots=True` on dataclasses, no stray TODOs.
- `config.json` stored alongside run outputs.
- `.ndjson` validated via `eval/schema_validator.py`; `pytest tests/test_schema.py` passes.
- Relevant CLIs smoke-tested.
- README/docs updated if behavior changed.
- Roadmap note: how work supports upcoming PRL/DMS additions.

## Out of Scope Unless Explicitly Requested

- Web/hosted leaderboard or evaluation services.
- GPU or non-CPU acceleration paths.
- Neural data fitting or spike-based models.
- Implementing PRL/DMS deliverables ahead of schedule without approval.
- Altering CLI surfaces, schema keys, or log formats.
