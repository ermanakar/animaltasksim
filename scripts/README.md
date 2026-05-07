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
`adaptive_control_interaction_sweep.py` is the current full-control arbitration
sweep; its strongest comparison candidate is `full_control_persist_half`, not a
new validated default.

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
