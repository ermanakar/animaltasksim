"""HTML report generation for AnimalTaskSim runs."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.metrics import compute_all_metrics, load_trials


def _encode_figure(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _psychometric_figure(df: pd.DataFrame, metrics: dict[str, object]) -> tuple[str, str]:
    if "stimulus_contrast" not in df.columns and "stimulus_coherence" not in df.columns:
        return "Psychometric", ""
    column = "stimulus_contrast" if "stimulus_contrast" in df.columns else "stimulus_coherence"
    grouped = df.groupby(column)["action"].apply(lambda x: (x == "right").mean()).reset_index()

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(grouped[column], grouped["action"], color="tab:blue", label="Observed")

    params = metrics.get("psychometric", {})
    if params:
        x = np.linspace(grouped[column].min(), grouped[column].max(), 100)
        slope = params.get("slope", np.nan)
        bias = params.get("bias", 0.0)
        lapse_low = params.get("lapse_low", 0.0)
        lapse_high = params.get("lapse_high", 0.0)
        if np.isfinite(slope):
            core = 1 / (1 + np.exp(-(x - bias) * slope))
            y = lapse_low + (1 - lapse_low - lapse_high) * core
            ax.plot(x, y, color="tab:orange", label="Fit")
    ax.set_xlabel(column.replace("stimulus_", ""))
    ax.set_ylabel("P(choice=right)")
    ax.set_title("Psychometric curve")
    ax.legend()
    return "Psychometric", _encode_figure(fig)


def _chronometric_figure(df: pd.DataFrame) -> tuple[str, str]:
    if "stimulus_coherence" not in df.columns or df["rt_ms"].isnull().all():
        return "Chronometric", ""
    grouped = df.groupby("stimulus_coherence")["rt_ms"].median().reset_index()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(grouped["stimulus_coherence"], grouped["rt_ms"], marker="o")
    ax.set_xlabel("|coherence|")
    ax.set_ylabel("Median RT (ms)")
    ax.set_title("Chronometric curve")
    return "Chronometric", _encode_figure(fig)


def _history_figure(df: pd.DataFrame, metrics: dict[str, object]) -> tuple[str, str]:
    params = metrics.get("history", {})
    if not params:
        return "History", ""
    labels = ["win_stay", "lose_shift", "sticky_choice"]
    values = [params.get(label, np.nan) for label in labels]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values, color=["tab:green", "tab:red", "tab:purple"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title("History metrics")
    return "History", _encode_figure(fig)


def build_report(log_path: Path, out_path: Path, *, title: str = "AnimalTaskSim Report", metrics: dict[str, object] | None = None) -> None:
    df = load_trials(log_path)
    if df.empty:
        raise ValueError(f"No trials found in log {log_path}")
    computed_metrics = metrics or compute_all_metrics(df, df["task"].iloc[0])

    figures: list[tuple[str, str]] = []
    figures.append(_psychometric_figure(df, computed_metrics))
    figures.append(_chronometric_figure(df))
    figures.append(_history_figure(df, computed_metrics))

    body_blocks = []
    for name, data_uri in figures:
        if not data_uri:
            continue
        body_blocks.append(
            f"<section><h2>{name}</h2><img src=\"data:image/png;base64,{data_uri}\" alt=\"{name}\"/></section>"
        )

    metrics_json = json.dumps(computed_metrics, indent=2)
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\"/>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; }}
            pre {{ background: #f4f4f4; padding: 1rem; }}
            img {{ max-width: 480px; border: 1px solid #ddd; padding: 4px; }}
            section {{ margin-bottom: 2rem; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p><strong>Log:</strong> {log_path}</p>
        {''.join(body_blocks)}
        <section>
            <h2>Metrics JSON</h2>
            <pre>{metrics_json}</pre>
        </section>
    </body>
    </html>
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


__all__ = ["build_report"]
