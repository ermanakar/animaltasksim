"""End-to-end run of the within-animal comparison on synthetic choices written in the converted-reference layout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from eval import within_animal as wa
from eval import within_animal_recovery as wr
from scripts import compare_within_animal as compare


def _converted(root: Path, cell: str = "drift_history", animals: int = 6, sessions: int = 6, trials: int = 160) -> Path:
    (root / "biased").mkdir(parents=True)
    manifest: dict = {"files": {}, "sessions": []}
    for index in range(animals):
        rng = np.random.default_rng(index)
        rows = [{"contrast": rng.choice(wa.CONTRASTS, trials), "prior": np.repeat([0.5, 0.2, 0.8, 0.2], trials // 4),
                 "choice": np.where(rng.random(trials) < 0.02, 0.0, 1.0), "correct": np.zeros(trials)} for _ in range(sessions)]
        data = wr.simulate(wa.pad(rows), cell, rng)[0]
        lines = []
        for s in range(sessions):
            eid = f"animal{index}-session{s}"
            manifest["sessions"].append({"id": eid, "subject": f"animal{index}", "lab": "lab_a" if index < 4 else "lab_b",
                                         "family": "biased", "status": "converted", "trials": trials, "session_number": sessions - s})
            for t in range(trials):
                action = {1.0: 0, -1.0: 1, 0.0: 2}[float(data.choice[s, t])]
                lines.append(json.dumps({"session_id": eid, "stimulus": {"contrast": float(data.contrast[s, t])},
                                         "block_prior": {"p_right": float(data.prior[s, t])}, "action": action,
                                         "correct": bool(data.correct[s, t])}))
        path = root / "biased" / f"animal{index}.ndjson"
        path.write_text("\n".join(lines) + "\n")
        manifest["files"][f"biased/animal{index}.ndjson"] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_loader_maps_actions_orders_sessions_and_checks_signs(tmp_path: Path) -> None:
    root = _converted(tmp_path / "converted", animals=1)
    lab, data = compare.load_sessions(root, 150, 6)["animal0"]
    first = [json.loads(line) for line in (root / "biased" / "animal0.ndjson").read_text().splitlines()[-160:]]
    assert lab == "lab_a" and data.choice.shape == (6, 160)
    assert np.array_equal(data.choice[0], [{0: 1.0, 1: -1.0, 2: 0.0}[r["action"]] for r in first])  # session_number orders
    assert not data.correct[data.choice == 0].any()
    flipped = (root / "biased" / "animal0.ndjson").read_text().replace('"action": 0', '"action": 9').replace(
        '"action": 1', '"action": 0').replace('"action": 9', '"action": 1')
    (root / "biased" / "animal0.ndjson").write_text(flipped)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["biased/animal0.ndjson"]["sha256"] = hashlib.sha256(flipped.encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="disagree"):
        compare.load_sessions(root, 150, 6)


def test_splits_cover_every_session_once() -> None:
    rng = np.random.default_rng(0)
    rows = [{"contrast": rng.choice(wa.CONTRASTS, 60), "prior": np.full(60, 0.5), "choice": np.ones(60), "correct": np.zeros(60)}
            for _ in range(5)]
    data = wr.simulate(wa.pad(rows), "history", rng)[0]
    alternating = compare.score(data, "alternating", 41, ("fixed",), 200)
    chronological = compare.score(data, "chronological", 41, ("fixed",), 200)
    assert alternating["trials"] == chronological["trials"] == int((data.choice != 0).sum())
    assert alternating["nll"]["fixed"] == wr.score_animal(data, models=("fixed",))["nll"]["fixed"]
    assert alternating["nll"]["fixed"] != chronological["nll"]["fixed"]
    with pytest.raises(ValueError):
        compare.score(data, "random", 41, ("fixed",), 200)


def test_plan_run_summarize_end_to_end_and_freezing(tmp_path: Path) -> None:
    root = _converted(tmp_path / "converted")
    args = lambda stage: compare.Args(stage, converted=root, output=tmp_path / "out", maxiter=60, resamples=200, workers=1)  # noqa: E731
    compare.run(args("plan"))
    with pytest.raises(RuntimeError, match="new empty"):
        compare.run(args("plan"))
    compare.run(args("run"))
    compare.run(args("summarize"))
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["primary"]["animals"] == 6 and result["sensitivity"]["without_largest_lab"]["animals"] == 2
    assert result["sensitivity"]["without_largest_lab"]["lab_removed"] == "lab_a"
    assert set(result["primary"]["gains"]) == set(wr.CONTRASTS_TESTED) | set(compare.DESCRIPTIVE)
    assert result["primary"]["gains"]["drift_total"]["mean"] > 0  # the synthetic animals drift
    assert (result["decisions"] is None) == (not result["conditions"]["met"])
    assert len((tmp_path / "out" / "subject_scores.csv").read_text().splitlines()) == 13
    with pytest.raises(RuntimeError, match="never replaced"):
        compare.run(args("summarize"))
    plan = tmp_path / "out" / "plan.json"
    plan.write_text(plan.read_text().replace('"maxiter": 60', '"maxiter": 61'))
    with pytest.raises(RuntimeError, match="Plan changed"):
        compare.run(args("run"))
