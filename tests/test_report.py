from __future__ import annotations

from agents.bayes_observer import BayesParams, BayesTrainingConfig, run_bayesian_observer
from eval.metrics import load_and_compute
from eval.report import build_report


def test_build_report_creates_html(tmp_path):
    output_dir = tmp_path / "bayes_run"
    config = BayesTrainingConfig(
        env="ibl_2afc",
        episodes=1,
        trials_per_episode=5,
        seed=101,
        output_dir=output_dir,
        params=BayesParams(sensory_sigma=0.3, lapse_rate=0.05),
    )
    run_bayesian_observer(config)
    log_path = config.output_paths()["log"]
    metrics = load_and_compute(log_path)

    out_path = tmp_path / "report.html"
    build_report(log_path, out_path, title="Test Report", metrics=metrics)

    html = out_path.read_text(encoding="utf-8")
    assert "Test Report" in html
    assert "Metrics JSON" in html
