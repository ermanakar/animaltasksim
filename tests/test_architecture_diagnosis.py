"""Checks for paired development diagnostic summaries."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.diagnose_architecture_fitting import bootstrap_delta


def test_pairs_by_subject_not_row_order() -> None:
    before = pd.Series({'a':1.,'b':2.,'c':None})
    after = pd.Series({'b':1.5,'a':.5,'c':9.})
    result = bootstrap_delta(before, after)
    assert result['mean_improvement'] == .5
    assert result['subject_bootstrap_95'] == [.5,.5]
    assert result['subjects'] == 2


def test_insufficient_paired_subjects_fails_closed() -> None:
    with pytest.raises(ValueError):
        bootstrap_delta(pd.Series({'a':1.}),pd.Series({'b':2.}))
