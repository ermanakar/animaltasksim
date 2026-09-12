"""Safety and protocol checks for the choice-only no-control comparison."""
from __future__ import annotations
from pathlib import Path

import pytest

from scripts.choice_control_comparison import Args, bootstrap_delta


def test_bootstrap_delta_reports_paired_improvement() -> None:
    import pandas as pd
    result = bootstrap_delta(pd.Series([2.0, 3.0, 4.0]), pd.Series([1.0, 2.0, 3.0]))
    assert result["mean_improvement"] == 1.0
    assert result["subjects"] == 3


def test_bootstrap_delta_rejects_single_pair() -> None:
    import pandas as pd
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_delta(pd.Series([1.0]), pd.Series([0.0]))


def test_runner_rejects_unmatched_settings_before_creating_output(tmp_path: Path) -> None:
    from scripts.choice_control_comparison import main
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Seeds and prediction samples"):
        main(Args(seeds=(7,), output=output))
    assert not output.exists()


def test_runner_refuses_nonempty_output(tmp_path: Path) -> None:
    (tmp_path / "existing").write_text("preserve")
    with pytest.raises(RuntimeError, match="new empty"):
        # Missing source is intentionally irrelevant: output protection comes first.
        from scripts.choice_control_comparison import main
        main(Args(source_run=tmp_path / "source", matched_run=tmp_path / "matched", output=tmp_path))
