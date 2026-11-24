"""The revision merges refitted switching scores into first-study animals and reproduces the first study's scoring."""
from __future__ import annotations

import numpy as np
import pytest

from eval import within_animal as wa
from eval import within_animal_recovery as wr
from scripts import recheck_within_animal_recovery as recheck


def _animal(**changes: object) -> dict:
    base = {"subject": "a", "trials": 100, "truth": {"bias": 0.1},
            "nll": {m: 0.5 for m in wa.MODELS}, "converged": {m: m != "switching_history" for m in wa.MODELS}}
    return base | changes


def test_merge_replaces_only_the_refitted_models() -> None:
    new = _animal(nll={"switching": 0.4, "switching_history": 0.3}, converged={"switching": True, "switching_history": True})
    merged = recheck.merge([_animal()], [new])[0]
    assert merged["nll"] == {m: 0.5 for m in recheck.CARRIED} | {"switching": 0.4, "switching_history": 0.3}
    assert all(merged["converged"].values())


def test_merge_rejects_different_data_and_unconverged_carried_fits() -> None:
    with pytest.raises(RuntimeError, match="differ"):
        recheck.merge([_animal()], [_animal(trials=99)])
    bad = _animal(converged={m: m != "drift" for m in wa.MODELS})
    with pytest.raises(RuntimeError, match="not converged"):
        recheck.merge([bad], [_animal()])


def test_score_matches_the_first_study_at_the_first_limit() -> None:
    rng = np.random.default_rng(0)
    rows = [{"contrast": rng.choice(wa.CONTRASTS, 80), "prior": np.full(80, 0.5), "choice": np.ones(80), "correct": np.zeros(80)}
            for _ in range(4)]
    data = wr.simulate(wa.pad(rows), "history", np.random.default_rng(1))[0]
    models = ("fixed", "history")
    assert recheck.score(data, models, maxiter=500)["nll"] == wr.score_animal(data, models=models)["nll"]
