# Archived Command Wrappers

This directory preserves one-off shell wrappers that documented specific
historical experiments. They are kept for provenance, not as supported CLI
entrypoints.

Supported and reusable entrypoints remain in `scripts/`, especially:

- `scripts/train_agent.py`
- `scripts/train_hybrid_curriculum.py`
- `scripts/train_adaptive_control.py`
- `scripts/train_r_ddm.py`
- `scripts/evaluate_agent.py`
- `scripts/make_report.py`
- `scripts/make_dashboard.py`
- sweep and validation Python scripts with current Tyro argument surfaces

The archived wrappers may contain older target values, paths, or experimental
assumptions. Before rerunning one, compare it against the current README,
FINDINGS, schema contract, and CLI help.
