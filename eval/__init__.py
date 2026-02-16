"""Evaluation and reporting utilities for AnimalTaskSim."""

from .metrics import compute_all_metrics, load_trials
from .report import build_report
from .schema_validator import TrialRecord, validate_file

__all__ = [
    "build_report",
    "compute_all_metrics",
    "load_trials",
    "TrialRecord",
    "validate_file",
]

# Dashboard is optional - import only if needed
try:
    from .dashboard import build_comparison_dashboard  # noqa: F401
    __all__.append("build_comparison_dashboard")
except ImportError:
    pass
