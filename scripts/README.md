# Script Entry Points

Keep `scripts/` for runnable Python entrypoints that are still useful against
the current schema, CLI contracts, and experiment registry workflow.

## Stable CLIs

- `train_agent.py`
- `train_hybrid_curriculum.py`
- `train_adaptive_control.py`
- `train_r_ddm.py`
- `evaluate_agent.py`
- `make_report.py`
- `make_dashboard.py`

Do not rename flags or output paths for these without explicit approval.

`train_adaptive_control.py` defaults to the recommended `persistence_only`
profile. Use `--control-profile full_control` only for explicitly labeled
comparison runs because exploration is experimental/unvalidated.
`adaptive_control_interaction_sweep.py` completed both the IBL bridge sweep and
the follow-up PRL arbitration sweep. That PRL result was negative and useful:
none of seven full-control scale variants preserved the exploration-only
block-learning curve. The follow-up sidecar diagnostic localized the deficit
to `uncertain_retry` firing after every failure under PRL's pinned perceptual
uncertainty.
`prl_arbitration_diagnostic.py` performs the sidecar step cheaply by rerolling
selected saved PRL checkpoints and writing an offline
`control_diagnostics.ndjson` sidecar. It does not change `trials.ndjson`.

`prl_transfer_validation_suite.py` runs the matched hidden-contingency transfer test:
five matched adaptive-control conditions in hidden-contingency probabilistic
reversal learning. PRL uses the IBL reference log to train the shared evidence
core, then evaluates zero-shot reward-driven adaptation. DMS remains an
environment scaffold and is intentionally not accepted by the adaptive-control
training CLI yet.

The May 30 matched run is complete under `runs/prl_transfer_validation_suite/`.
Read `prl_block_learning_lift` alongside the original 10-trial
`prl_adaptation_lift`: the exploration-only lesion learns slowly across each
hidden-contingency block, and the shorter window under-reports that recovery.
The follow-up PRL interaction sweep is complete under
`runs/prl_adaptive_control_interaction_sweep_v1/`: 50 usable runs and 80,000
schema-valid trials. The flag-gated change-evidence recurrence then tested the
state-dependent fix. Safety-gated calibration rejected λ=0.7 as too eager and
selected λ=0.9 as the validated opt-in cross-task profile: with `uncertain_retry`
still enabled, full control reaches PRL block-learning lift `+0.469` and
optimal choice `0.706`; after the June 1 prior-trial retry-metric correction,
its IBL retry gap is `0.158` versus the historical flag-off `0.175`. The feature
remains default off.

## Data Acquisition

`fetch_ibl_reference.py` expands the animal reference set beyond the current
10-session `data/ibl/reference.ndjson` by pulling `biasedChoiceWorld` sessions
from the IBL public server (OpenAlyx, anonymous) into the project schema. It
requires `ONE-api` (`pip install ONE-api`) installed in a throwaway env — it is
deliberately kept out of `pyproject.toml`. It derives each action from the
stimulus side and `feedbackType` (convention-agnostic), auto-calibrates the IBL
`choice` sign for zero-contrast trials and reports its agreement, filters to the
biased-blocks contrast set (`0.5` excluded), and defaults RT to
`firstMovement_times - stimOn_times` to match the existing reference. It applies
a trained-performance QC gate (`--min-easy-accuracy`, default 0.85 on
full-contrast trials) and shuffles the candidate list deterministically, because
the Alyx search order front-loads early/low-performance sessions. It writes a
manifest of session EIDs and does **not** overwrite `reference.ndjson`.
Compare targets with `compute_reference_targets.py` before adopting anything.

## Sweep And Validation Scripts

Sweep scripts may encode a specific scientific hypothesis, but shared mechanics
belong in `_sweep_utils.py` rather than being copied between files. Prefer this
shape for new sweeps:

- typed `@dataclass(slots=True)` arguments
- `tyro.cli(...)`
- `run_root` under `runs/`
- per-run `config.json`, `trials.ndjson`, and `metrics.json`
- `sweep_summary.csv` or a named CSV/JSON summary
- explicit notes in `FINDINGS.md` when the result changes the research state

Historical shell wrappers belong in `docs/archive/commands/`.
