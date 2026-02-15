"""Generate standalone PNG figures for the README from K2 experiment results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import compute_all_metrics, load_trials


def _load_data(
    agent_path: str, reference_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    df_agent = load_trials(agent_path)
    df_ref = load_trials(reference_path)
    # Infer task from the stimulus column present
    task = "rdm" if "stimulus_coherence" in df_agent.columns else "ibl_2afc"
    metrics_agent = compute_all_metrics(df_agent, task=task)
    metrics_ref = compute_all_metrics(df_ref, task=task)
    return df_agent, df_ref, metrics_agent, metrics_ref


def _sigmoid(x: np.ndarray, slope: float, bias: float, ll: float, lh: float) -> np.ndarray:
    core = 1 / (1 + np.exp(-(x - bias) * slope))
    return ll + (1 - ll - lh) * core


def generate_combined_figure(
    agent_path: str,
    reference_path: str,
    output_path: str,
    agent_name: str = "Hybrid DDM+LSTM",
    ref_name: str = "Macaque (Roitman & Shadlen)",
) -> None:
    """Generate a 2x2 comparison figure with psychometric, chronometric,
    accuracy, and history effects panels."""
    df_agent, df_ref, m_agent, m_ref = _load_data(agent_path, reference_path)

    # Determine stimulus column
    stim_col = (
        "stimulus_coherence" if "stimulus_coherence" in df_agent.columns else "stimulus_contrast"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "AnimalTaskSim: Agent vs Macaque Behavioral Fingerprints",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    AGENT_COLOR = "#2196F3"
    REF_COLOR = "#4CAF50"

    # ── Panel A: Psychometric ──────────────────────────────────
    ax = axes[0, 0]
    g_a = df_agent.groupby(stim_col)["action"].apply(lambda x: (x == "right").mean()).reset_index()
    g_r = df_ref.groupby(stim_col)["action"].apply(lambda x: (x == "right").mean()).reset_index()

    ax.scatter(g_a[stim_col], g_a["action"], color=AGENT_COLOR, s=70, alpha=0.8, label=agent_name, zorder=3)
    ax.scatter(g_r[stim_col], g_r["action"], color=REF_COLOR, s=70, alpha=0.8, label=ref_name, marker="s", zorder=3)

    x_fit = np.linspace(
        min(g_a[stim_col].min(), g_r[stim_col].min()),
        max(g_a[stim_col].max(), g_r[stim_col].max()),
        200,
    )
    for params, color, ls in [
        (m_agent.get("psychometric", {}), AGENT_COLOR, "-"),
        (m_ref.get("psychometric", {}), REF_COLOR, "--"),
    ]:
        if params and np.isfinite(params.get("slope", np.nan)):
            y_fit = _sigmoid(
                x_fit, params["slope"], params.get("bias", 0),
                params.get("lapse_low", 0), params.get("lapse_high", 0),
            )
            ax.plot(x_fit, y_fit, color=color, linewidth=2.5, linestyle=ls, alpha=0.8)

    ax.set_xlabel("Signed Coherence", fontsize=11)
    ax.set_ylabel("P(Choose Right)", fontsize=11)
    ax.set_title("A. Psychometric Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # ── Panel B: Chronometric ──────────────────────────────────
    ax = axes[0, 1]
    for df, color, marker, ls, label in [
        (df_agent, AGENT_COLOR, "o", "-", agent_name),
        (df_ref, REF_COLOR, "s", "--", ref_name),
    ]:
        df_tmp = df.copy()
        df_tmp["difficulty"] = df_tmp[stim_col].abs()
        grp = df_tmp.groupby("difficulty").agg(
            rt_mean=("rt_ms", "mean"), rt_std=("rt_ms", "std"), n=("rt_ms", "count")
        ).reset_index()
        grp["rt_sem"] = grp["rt_std"] / np.sqrt(grp["n"])
        ax.errorbar(
            grp["difficulty"], grp["rt_mean"], yerr=grp["rt_sem"],
            fmt=f"{marker}{ls}", color=color, linewidth=2, markersize=8,
            capsize=5, label=label, alpha=0.8,
        )

    ax.set_xlabel("|Coherence|", fontsize=11)
    ax.set_ylabel("Mean RT (ms)", fontsize=11)
    ax.set_title("B. Chronometric Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel C: Accuracy ──────────────────────────────────────
    ax = axes[1, 0]
    for df, color, marker, ls, label in [
        (df_agent, AGENT_COLOR, "o", "-", agent_name),
        (df_ref, REF_COLOR, "s", "--", ref_name),
    ]:
        df_tmp = df.copy()
        df_tmp["difficulty"] = df_tmp[stim_col].abs()
        grp = df_tmp.groupby("difficulty")["correct"].mean().reset_index()
        ax.plot(
            grp["difficulty"], grp["correct"], f"{marker}{ls}",
            color=color, linewidth=2, markersize=8, label=label, alpha=0.8,
        )

    ax.set_xlabel("|Coherence|", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("C. Accuracy vs Difficulty", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    # ── Panel D: Key Metrics Summary ───────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    labels = ["Psych Slope", "Psych Bias", "Chrono Slope\n(ms/unit)", "Win-Stay", "Lose-Shift"]
    pa = m_agent.get("psychometric", {})
    ca = m_agent.get("chronometric", {})
    ha = m_agent.get("history", {})
    pr = m_ref.get("psychometric", {})
    cr = m_ref.get("chronometric", {})
    hr = m_ref.get("history", {})

    vals_agent = [
        pa.get("slope", np.nan),
        pa.get("bias", np.nan),
        ca.get("slope_ms_per_unit", np.nan),
        ha.get("win_stay", np.nan),
        ha.get("lose_shift", np.nan),
    ]
    vals_ref = [
        pr.get("slope", np.nan),
        pr.get("bias", np.nan),
        cr.get("slope_ms_per_unit", np.nan),
        hr.get("win_stay", np.nan),
        hr.get("lose_shift", np.nan),
    ]

    x = np.arange(len(labels))
    width = 0.35

    # Normalize for display (different scales)
    table_data = []
    for i, lbl in enumerate(labels):
        va = vals_agent[i]
        vr = vals_ref[i]
        va_str = f"{va:.2f}" if np.isfinite(va) else "N/A"
        vr_str = f"{vr:.2f}" if np.isfinite(vr) else "N/A"
        table_data.append([lbl, va_str, vr_str])

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", agent_name, ref_name],
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.3, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    # Style header
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#e0e0e0")
        cell.set_text_props(fontweight="bold")
    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            cell = table[i, j]
            cell.set_facecolor("#f9f9f9" if i % 2 == 0 else "white")

    ax.set_title("D. Metrics Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    agent_log = root / "runs" / "decoupling_K2_window_control" / "trials.ndjson"
    ref_log = root / "data" / "macaque" / "reference.ndjson"
    output = root / "docs" / "figures" / "k2_dashboard.png"

    if not agent_log.exists():
        print(f"Agent log not found: {agent_log}")
        sys.exit(1)
    if not ref_log.exists():
        print(f"Reference log not found: {ref_log}")
        sys.exit(1)

    generate_combined_figure(
        str(agent_log),
        str(ref_log),
        str(output),
    )
    print(f"\nFigure ready for README at: {output.relative_to(root)}")
