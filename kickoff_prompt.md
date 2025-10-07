You are a senior coding AI tasked with scaffolding **AnimalTaskSim** — a Python 3.11 project that reproduces two animal behavioral tasks as Gymnasium environments, trains baseline agents, and evaluates them with animal-style metrics.

## Ground rules
- Language: Python 3.11, type hints, docstrings, no notebooks for core code.
- Packages: gymnasium, numpy, scipy, pandas, torch, stable-baselines3, matplotlib, pydantic, pytest, tyro (or argparse).
- Reproducibility: global seeding, config objects saved to disk.
- I/O: **All** training/eval writes newline-delimited JSON (`.ndjson`) per trial matching the unified schema below.
- Keep CPU-only by default; runs must complete demo configs < 20 minutes.

## Unified trial-log schema (.ndjson per trial)
```
task, session_id, trial_index, stimulus({contrast/side or coherence/dir}), block_prior, action, correct, reward, rt_ms, phase_times, prev, seed, agent({name,version})
```

## Files to create
```
envs/ibl_2afc.py            # Mouse 2AFC env
envs/rdm_macaque.py         # RDM env
envs/utils_timing.py        # Phase & timing helpers
agents/sticky_q.py          # Q-learning + stickiness baseline
agents/bayes_observer.py    # Ideal observer with sensory noise + lapses
agents/ppo_baseline.py      # SB3 PPO wrapper with action masking
eval/metrics.py             # psychometric, chronometric, history kernels
eval/fitters.py             # logistic fits, RT fits
eval/report.py              # HTML report generation (matplotlib figs)
scripts/train_agent.py      # CLI to train agents; writes .ndjson logs
scripts/evaluate_agent.py   # CLI to compute metrics from logs
scripts/make_report.py      # CLI to render HTML report from metrics
tests/test_envs.py
tests/test_metrics.py
tests/test_agents.py
pyproject.toml              # build metadata & deps
```

## Env specifications

### envs/ibl_2afc.py
- Observation: dict space
  - `contrast` in [-1,1] (sign = side; |value| = difficulty)
  - optional `phase_onehot` and `t_norm`
- Action space: Discrete(3) = {left, right, no-op}
- Reward: +1 correct; 0 otherwise; configurable timeouts & ITI.
- Episode: fixed N trials per session.
- Config: contrast set, block schedule, timings, reward sizes.
- `step()` must advance phases and enforce response windows.
- Log each trial to `.ndjson` in `close()` or via a logger.

### envs/rdm_macaque.py
- Observation: dict
  - `coherence` ∈ [0,1] signed by direction
  - optional `phase_onehot`, `t_norm`
- Action space: Discrete(3) = {left, right, hold}
- Reward: +1 correct; optional per-step cost (speed–accuracy tradeoff)
- Optional collapsing bound via time-varying decision rule / termination.
- Same logging requirements.

## Baseline agents

### agents/sticky_q.py
- Tabular Q-learning with softmax policy and **stickiness term** β_stay that biases repeating last action.
- Train loop independent of SB3; interacts via Gymnasium API.
- Outputs `.ndjson` using the schema.

### agents/bayes_observer.py
- Encodes belief about side given noisy observation (contrast/coherence) with Gaussian sensory noise σ.
- Lapse parameter ε; softmax over posterior; logs trials to `.ndjson`.

### agents/ppo_baseline.py
- SB3 PPO with action masking so only valid actions during response phase.
- Small network, small batch sizes; seeds fixed; logs via a wrapper that writes `.ndjson` after each trial.

## Evaluation

### eval/metrics.py
- **Psychometric**: fit logistic to P(choice=right) vs signed stimulus.
- **Chronometric (RDM)**: median RT vs coherence; report slope and intercept.
- **History**: win-stay/lose-shift; sticky-choice; logistic regression kernel over past K trials.
- **Bias**: block-prior effect size on choice bias.

### eval/report.py
- Generate HTML with matplotlib plots: curves + tables; embed JSON metrics.

## CLIs

### scripts/train_agent.py
- Args: `--env {ibl_2afc,rdm}`, `--agent {sticky_q,bayes,ppo}`, `--steps`, `--out`, `--seed`, config overrides.
- Saves config.json, seeds, and logs.

### scripts/evaluate_agent.py
- Args: `--run` path to logs; outputs metrics.json.

### scripts/make_report.py
- Args: `--run` and `--out` path; produces `report.html`.

## Tests (pytest)
- `test_envs.py`: reset/step invariants, phase transitions, action masking.
- `test_metrics.py`: deterministic toy data → known metric values.
- `test_agents.py`: smoke tests that produce logs with required fields.

## Acceptance criteria (for this sprint)
- `make demo_ibl` trains `sticky_q` on `ibl_2afc` (≤5 minutes) and produces a report with a sensible psychometric curve and non-zero stickiness.
- `make demo_rdm` trains `ppo` on `rdm` and shows a speed–accuracy tradeoff when per-step cost > 0.
- All tests pass on CI (GitHub Actions matrix: py3.11, ubuntu-latest).

## Deliver next
- Implement the above with clean, documented code. Avoid over-design; keep functions small and pure where possible. Save configs, keep seeds constant, write logs.
