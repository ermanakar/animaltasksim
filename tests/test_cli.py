from __future__ import annotations

import json

from scripts.evaluate_agent import EvaluateArgs, main as evaluate_main


def _write_log(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "task": "ibl_2afc",
        "session_id": "session-123",
        "trial_index": 0,
        "stimulus": {"contrast": 0.25, "side": "right"},
        "block_prior": {"p_right": 0.5},
        "action": "right",
        "correct": True,
        "reward": 1.0,
        "rt_ms": 500.0,
        "phase_times": {"stim_ms": 300, "resp_ms": 700},
        "prev": {"action": "left", "reward": 0.0, "correct": False},
        "seed": 1,
        "agent": {"name": "tester", "version": "0.0"},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_evaluate_cli_with_run(tmp_path):
    run_dir = tmp_path / "run"
    log_path = run_dir / "trials.ndjson"
    _write_log(log_path)

    args = EvaluateArgs(run=run_dir)
    evaluate_main(args)

    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["log"].endswith("trials.ndjson")


def test_evaluate_cli_with_log_only(tmp_path):
    log_path = tmp_path / "reference.ndjson"
    _write_log(log_path)

    args = EvaluateArgs(run=None, log=log_path, out=None)
    evaluate_main(args)

    metrics_path = log_path.parent / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["log"].endswith("reference.ndjson")
