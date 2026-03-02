"""Interactive comparison dashboard for agent vs animal behavioral fingerprints."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from eval.metrics import compute_all_metrics, load_trials

# Color palette matching README figures
_COLOR_AGENT = "#2166ac"
_COLOR_REFERENCE = "#4d4d4d"


def _encode_figure(fig: Figure) -> str:
    """Encode matplotlib figure as base64 PNG."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _compute_per_session_stats(
    df: pd.DataFrame, task: str
) -> dict[str, dict[str, float]] | None:
    """Compute per-session reference statistics.

    Groups reference data by session_id, computes metrics per session,
    and returns summary stats (mean, std, n) for key metrics.

    Returns None if fewer than 2 sessions (e.g., macaque data).
    """
    session_ids = df["session_id"].unique()
    if len(session_ids) < 2:
        return None

    per_session: dict[str, list[float]] = {
        "psych_slope": [],
        "chrono_slope": [],
        "win_stay": [],
        "lose_shift": [],
        "sticky_choice": [],
        "psych_bias": [],
        "lapse_low": [],
        "lapse_high": [],
    }

    for sid in session_ids:
        session_df = df[df["session_id"] == sid]
        metrics = compute_all_metrics(session_df, task)

        psych = metrics.get("psychometric", {})
        chrono = metrics.get("chronometric", {})
        history = metrics.get("history", {})

        slope_val = psych.get("slope") if isinstance(psych, dict) else None
        if slope_val is not None and np.isfinite(slope_val):
            # Filter degenerate sessions (psych_slope > 100)
            if slope_val <= 100:
                per_session["psych_slope"].append(slope_val)

        bias_val = psych.get("bias") if isinstance(psych, dict) else None
        if bias_val is not None and np.isfinite(bias_val):
            per_session["psych_bias"].append(bias_val)

        lapse_low_val = psych.get("lapse_low") if isinstance(psych, dict) else None
        if lapse_low_val is not None and np.isfinite(lapse_low_val):
            per_session["lapse_low"].append(lapse_low_val)

        lapse_high_val = psych.get("lapse_high") if isinstance(psych, dict) else None
        if lapse_high_val is not None and np.isfinite(lapse_high_val):
            per_session["lapse_high"].append(lapse_high_val)

        chrono_val = chrono.get("slope_ms_per_unit") if isinstance(chrono, dict) else None
        if chrono_val is not None and np.isfinite(chrono_val):
            per_session["chrono_slope"].append(chrono_val)

        ws_val = history.get("win_stay") if isinstance(history, dict) else None
        if ws_val is not None and np.isfinite(ws_val):
            per_session["win_stay"].append(ws_val)

        ls_val = history.get("lose_shift") if isinstance(history, dict) else None
        if ls_val is not None and np.isfinite(ls_val):
            per_session["lose_shift"].append(ls_val)

        sc_val = history.get("sticky_choice") if isinstance(history, dict) else None
        if sc_val is not None and np.isfinite(sc_val):
            per_session["sticky_choice"].append(sc_val)

    result: dict[str, dict[str, float]] = {}
    for key, values in per_session.items():
        if len(values) >= 2:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "n": len(values),
            }

    return result if result else None


def _compute_sigma_match(
    agent_val: float | None,
    mean: float,
    std: float,
) -> tuple[str, str, str]:
    """Compute sigma-based match indicator.

    Returns (html_string, css_class, sigma_value).
    """
    if agent_val is None or not np.isfinite(agent_val):
        return ("", "", "")
    if std == 0 or not np.isfinite(std):
        return ("", "", "")

    z = abs(agent_val - mean) / std

    if z <= 1.0:
        css_class = "match-green"
        symbol = "&#10003;"  # checkmark
    elif z <= 2.0:
        css_class = "match-yellow"
        symbol = "~"
    else:
        css_class = "match-red"
        symbol = "&#10007;"  # cross

    html = f'<span class="match-badge {css_class}">{z:.1f}&sigma; {symbol}</span>'
    return (html, css_class, f"{z:.1f}")


def _compute_match_percentage(agent_val: float | None, animal_val: float | None) -> str:
    """Compute match percentage between agent and animal values (fallback)."""
    if agent_val is None or animal_val is None:
        return ""
    if not np.isfinite(agent_val) or not np.isfinite(animal_val):
        return ""
    if animal_val == 0:
        return ""

    ratio = agent_val / animal_val
    percentage = ratio * 100

    if 90 <= percentage <= 110:
        css_class = "match-green"
        emoji = "&#10003;"
    elif 80 <= percentage <= 120:
        css_class = "match-yellow"
        emoji = "~"
    else:
        css_class = "match-red"
        emoji = "&#10007;"

    return f'<span class="match-badge {css_class}">{percentage:.1f}% {emoji}</span>'


def _plot_psychometric_comparison(
    ax: Axes,
    df_agent: pd.DataFrame,
    df_animal: pd.DataFrame,
    metrics_agent: dict[str, Any],
    metrics_animal: dict[str, Any],
    column: str,
) -> None:
    """Plot psychometric curves side-by-side."""
    # Agent data
    grouped_agent = df_agent.groupby(column)["action"].apply(
        lambda x: (x == "right").mean()
    ).reset_index()
    ax.scatter(
        grouped_agent[column],
        grouped_agent["action"],
        color=_COLOR_AGENT,
        s=80,
        alpha=0.7,
        label="Agent data",
        marker="o",
        zorder=3,
    )

    # Animal data
    grouped_animal = df_animal.groupby(column)["action"].apply(
        lambda x: (x == "right").mean()
    ).reset_index()
    ax.scatter(
        grouped_animal[column],
        grouped_animal["action"],
        color=_COLOR_REFERENCE,
        s=80,
        alpha=0.7,
        label="Animal data",
        marker="s",
        zorder=3,
    )

    # Fitted curves
    x_range = np.linspace(
        min(grouped_agent[column].min(), grouped_animal[column].min()),
        max(grouped_agent[column].max(), grouped_animal[column].max()),
        200,
    )

    # Agent fit
    params_agent = metrics_agent.get("psychometric", {})
    if params_agent and np.isfinite(params_agent.get("slope", np.nan)):
        slope = params_agent["slope"]
        bias = params_agent.get("bias", 0.0)
        lapse_low = params_agent.get("lapse_low", 0.0)
        lapse_high = params_agent.get("lapse_high", 0.0)
        core = 1 / (1 + np.exp(-(x_range - bias) * slope))
        y = lapse_low + (1 - lapse_low - lapse_high) * core
        ax.plot(x_range, y, color=_COLOR_AGENT, linewidth=2.5, label="Agent fit", alpha=0.8)

    # Animal fit
    params_animal = metrics_animal.get("psychometric", {})
    if params_animal and np.isfinite(params_animal.get("slope", np.nan)):
        slope = params_animal["slope"]
        bias = params_animal.get("bias", 0.0)
        lapse_low = params_animal.get("lapse_low", 0.0)
        lapse_high = params_animal.get("lapse_high", 0.0)
        core = 1 / (1 + np.exp(-(x_range - bias) * slope))
        y = lapse_low + (1 - lapse_low - lapse_high) * core
        ax.plot(
            x_range, y, color=_COLOR_REFERENCE, linewidth=2.5,
            label="Animal fit", alpha=0.8, linestyle="--",
        )

    ax.set_xlabel(column.replace("stimulus_", "").capitalize(), fontsize=11)
    ax.set_ylabel("P(choice = right)", fontsize=11)
    ax.set_title("Psychometric Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


def _plot_chronometric_comparison(
    ax: Axes,
    df_agent: pd.DataFrame,
    df_animal: pd.DataFrame,
    metrics_agent: dict[str, Any],
    metrics_animal: dict[str, Any],
) -> None:
    """Plot chronometric curves (RT vs coherence/contrast)."""
    # Determine stimulus column
    if "stimulus_coherence" in df_agent.columns:
        stim_col = "stimulus_coherence"
        xlabel = "|Coherence|"
    elif "stimulus_contrast" in df_agent.columns:
        stim_col = "stimulus_contrast"
        xlabel = "|Contrast|"
    else:
        ax.text(0.5, 0.5, "No stimulus data", ha="center", va="center", transform=ax.transAxes)
        return

    # Use absolute values for difficulty
    df_agent = df_agent.copy()
    df_animal = df_animal.copy()
    df_agent["difficulty"] = df_agent[stim_col].abs()
    df_animal["difficulty"] = df_animal[stim_col].abs()

    # Agent RT by difficulty
    grouped_agent = df_agent.groupby("difficulty").agg(
        rt_mean=("rt_ms", "mean"),
        rt_std=("rt_ms", "std"),
        rt_count=("rt_ms", "count"),
    ).reset_index()
    grouped_agent["rt_sem"] = grouped_agent["rt_std"] / np.sqrt(grouped_agent["rt_count"])

    # Animal RT by difficulty
    grouped_animal = df_animal.groupby("difficulty").agg(
        rt_mean=("rt_ms", "mean"),
        rt_std=("rt_ms", "std"),
        rt_count=("rt_ms", "count"),
    ).reset_index()
    grouped_animal["rt_sem"] = grouped_animal["rt_std"] / np.sqrt(grouped_animal["rt_count"])

    # Plot agent
    ax.errorbar(
        grouped_agent["difficulty"],
        grouped_agent["rt_mean"],
        yerr=grouped_agent["rt_sem"],
        fmt="o-",
        color=_COLOR_AGENT,
        linewidth=2,
        markersize=8,
        capsize=5,
        label="Agent",
        alpha=0.8,
    )

    # Plot animal
    ax.errorbar(
        grouped_animal["difficulty"],
        grouped_animal["rt_mean"],
        yerr=grouped_animal["rt_sem"],
        fmt="s--",
        color=_COLOR_REFERENCE,
        linewidth=2,
        markersize=8,
        capsize=5,
        label="Animal",
        alpha=0.8,
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Mean RT (ms)", fontsize=11)
    ax.set_title("Chronometric Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)


def _plot_history_comparison(
    ax: Axes,
    metrics_agent: dict[str, Any],
    metrics_animal: dict[str, Any],
) -> None:
    """Plot history effects comparison."""
    history_agent = metrics_agent.get("history", {})
    history_animal = metrics_animal.get("history", {})

    if not history_agent or not history_animal:
        ax.text(0.5, 0.5, "No history data", ha="center", va="center", transform=ax.transAxes)
        return

    labels = ["Win-stay", "Lose-shift", "Sticky-choice"]
    metrics_keys = ["win_stay", "lose_shift", "sticky_choice"]

    values_agent = [history_agent.get(k, 0) for k in metrics_keys]
    values_animal = [history_animal.get(k, 0) for k in metrics_keys]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, values_agent, width, label="Agent", color=_COLOR_AGENT, alpha=0.8)
    ax.bar(x + width / 2, values_animal, width, label="Animal", color=_COLOR_REFERENCE, alpha=0.8)

    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title("History Effects", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_accuracy_by_coherence(
    ax: Axes,
    df_agent: pd.DataFrame,
    df_animal: pd.DataFrame,
) -> None:
    """Plot accuracy by coherence/contrast level."""
    # Determine stimulus column
    if "stimulus_coherence" in df_agent.columns:
        stim_col = "stimulus_coherence"
        xlabel = "|Coherence|"
    elif "stimulus_contrast" in df_agent.columns:
        stim_col = "stimulus_contrast"
        xlabel = "|Contrast|"
    else:
        ax.text(0.5, 0.5, "No stimulus data", ha="center", va="center", transform=ax.transAxes)
        return

    # Use absolute values for difficulty
    df_agent = df_agent.copy()
    df_animal = df_animal.copy()
    df_agent["difficulty"] = df_agent[stim_col].abs()
    df_animal["difficulty"] = df_animal[stim_col].abs()

    # Agent accuracy
    grouped_agent = df_agent.groupby("difficulty")["correct"].mean().reset_index()
    # Animal accuracy
    grouped_animal = df_animal.groupby("difficulty")["correct"].mean().reset_index()

    ax.plot(
        grouped_agent["difficulty"],
        grouped_agent["correct"],
        "o-",
        color=_COLOR_AGENT,
        linewidth=2,
        markersize=8,
        label="Agent",
        alpha=0.8,
    )
    ax.plot(
        grouped_animal["difficulty"],
        grouped_animal["correct"],
        "s--",
        color=_COLOR_REFERENCE,
        linewidth=2,
        markersize=8,
        label="Animal",
        alpha=0.8,
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"Accuracy vs {xlabel.replace('|', '')}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)


def _fmt_value(val: float | None, fmt: str = ".2f") -> str:
    """Format a metric value, handling None/NaN."""
    if val is None or not np.isfinite(val):
        return "—"
    return f"{val:{fmt}}"


def _fmt_per_session(stats: dict[str, float], fmt: str = ".2f") -> str:
    """Format per-session stats as 'mean +/- std (n=N)'."""
    mean = stats["mean"]
    std = stats["std"]
    n = int(stats["n"])
    return f"{mean:{fmt}} &plusmn; {std:{fmt}} (n={n})"


def _create_metrics_table_html(
    metrics_agent: dict[str, Any],
    metrics_animal: dict[str, Any],
    per_session_stats: dict[str, dict[str, float]] | None = None,
) -> str:
    """Generate HTML table comparing key metrics with per-session reference stats."""
    psych_agent = metrics_agent.get("psychometric", {})
    psych_animal = metrics_animal.get("psychometric", {})
    chrono_agent = metrics_agent.get("chronometric", {})
    chrono_animal = metrics_animal.get("chronometric", {})
    hist_agent = metrics_agent.get("history", {})
    hist_animal = metrics_animal.get("history", {})

    # Row definition: (label, agent_val, animal_aggregate, per_session_key, fmt, is_key_metric)
    rows: list[tuple[str, float | None, float | None, str | None, str, bool]] = []

    # Key metrics first
    rows.append((
        "Psychometric Slope",
        psych_agent.get("slope") if isinstance(psych_agent, dict) else None,
        psych_animal.get("slope") if isinstance(psych_animal, dict) else None,
        "psych_slope", ".2f", True,
    ))
    rows.append((
        "Chrono Slope (ms/unit)",
        chrono_agent.get("slope_ms_per_unit") if isinstance(chrono_agent, dict) else None,
        chrono_animal.get("slope_ms_per_unit") if isinstance(chrono_animal, dict) else None,
        "chrono_slope", ".1f", True,
    ))
    rows.append((
        "Win-stay",
        hist_agent.get("win_stay") if isinstance(hist_agent, dict) else None,
        hist_animal.get("win_stay") if isinstance(hist_animal, dict) else None,
        "win_stay", ".3f", True,
    ))
    rows.append((
        "Lose-shift",
        hist_agent.get("lose_shift") if isinstance(hist_agent, dict) else None,
        hist_animal.get("lose_shift") if isinstance(hist_animal, dict) else None,
        "lose_shift", ".3f", True,
    ))

    # Secondary metrics
    rows.append((
        "Psychometric Bias",
        psych_agent.get("bias") if isinstance(psych_agent, dict) else None,
        psych_animal.get("bias") if isinstance(psych_animal, dict) else None,
        "psych_bias", ".3f", False,
    ))
    rows.append((
        "Lapse (low)",
        psych_agent.get("lapse_low") if isinstance(psych_agent, dict) else None,
        psych_animal.get("lapse_low") if isinstance(psych_animal, dict) else None,
        "lapse_low", ".3f", False,
    ))
    rows.append((
        "Lapse (high)",
        psych_agent.get("lapse_high") if isinstance(psych_agent, dict) else None,
        psych_animal.get("lapse_high") if isinstance(psych_animal, dict) else None,
        "lapse_high", ".3f", False,
    ))
    if chrono_agent and chrono_animal:
        rows.append((
            "Chrono Intercept (ms)",
            chrono_agent.get("intercept_ms") if isinstance(chrono_agent, dict) else None,
            chrono_animal.get("intercept_ms") if isinstance(chrono_animal, dict) else None,
            None, ".1f", False,
        ))
    rows.append((
        "Sticky-choice",
        hist_agent.get("sticky_choice") if isinstance(hist_agent, dict) else None,
        hist_animal.get("sticky_choice") if isinstance(hist_animal, dict) else None,
        "sticky_choice", ".3f", False,
    ))

    table_html = """
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Agent</th>
                <th>Animal</th>
                <th>Match</th>
            </tr>
        </thead>
        <tbody>
    """

    for label, agent_val, animal_agg, ps_key, fmt, is_key in rows:
        border_class = "key-metric" if is_key else "secondary-metric"

        # Agent value
        agent_str = _fmt_value(agent_val, fmt)

        # Animal value: prefer per-session stats if available
        stats = per_session_stats.get(ps_key) if per_session_stats and ps_key else None
        if stats:
            animal_str = _fmt_per_session(stats, fmt)
        else:
            animal_str = _fmt_value(animal_agg, fmt)

        # Match indicator: sigma-based if per-session available, else ratio fallback
        if stats and agent_val is not None and np.isfinite(agent_val):
            match_html, _, _ = _compute_sigma_match(agent_val, stats["mean"], stats["std"])
        elif is_key:
            match_html = _compute_match_percentage(agent_val, animal_agg)
        else:
            match_html = ""

        match_cell = f'<td class="match-cell">{match_html}</td>' if match_html else "<td></td>"
        table_html += f"""
            <tr class="{border_class}">
                <td><strong>{label}</strong></td>
                <td>{agent_str}</td>
                <td>{animal_str}</td>
                {match_cell}
            </tr>
        """

    table_html += """
        </tbody>
    </table>
    <div class="sigma-legend">
        <span class="match-badge match-green">&#10003;</span> Within 1&sigma;
        &nbsp;&nbsp;
        <span class="match-badge match-yellow">~</span> Within 2&sigma;
        &nbsp;&nbsp;
        <span class="match-badge match-red">&#10007;</span> Outside 2&sigma;
    </div>
    """
    return table_html


def build_comparison_dashboard(
    agent_log_path: Path,
    animal_log_path: Path,
    output_path: Path,
    *,
    title: str = "Agent vs Animal Comparison",
    agent_name: str = "Agent",
    animal_name: str = "Animal",
    animal_metrics_override: dict[str, Any] | None = None,
) -> None:
    """
    Build an interactive HTML dashboard comparing agent and animal behavior.

    Parameters
    ----------
    agent_log_path : Path
        Path to agent's NDJSON log file.
    animal_log_path : Path
        Path to animal's NDJSON reference file.
    output_path : Path
        Path to save the HTML dashboard.
    title : str
        Dashboard title.
    agent_name : str
        Name to display for agent (e.g., "Sticky-GLM v18").
    animal_name : str
        Name to display for animal (e.g., "IBL Reference").
    """
    # Load data
    df_agent = load_trials(agent_log_path)
    df_animal = load_trials(animal_log_path)

    if df_agent.empty:
        raise ValueError(f"No trials in agent log: {agent_log_path}")
    if df_animal.empty:
        raise ValueError(f"No trials in animal log: {animal_log_path}")

    # Compute metrics
    task_name = df_agent["task"].iloc[0]
    metrics_agent = compute_all_metrics(df_agent, task_name)
    metrics_animal = compute_all_metrics(df_animal, task_name)
    if animal_metrics_override:
        for key, value in animal_metrics_override.items():
            if isinstance(value, dict) and isinstance(metrics_animal.get(key), dict):
                metrics_animal[key].update(value)  # type: ignore[arg-type]
            else:
                metrics_animal[key] = value

    # Compute per-session reference statistics
    per_session_stats = _compute_per_session_stats(df_animal, task_name)
    n_sessions = df_animal["session_id"].nunique()

    # Determine stimulus column
    if "stimulus_contrast" in df_agent.columns:
        stim_column = "stimulus_contrast"
    elif "stimulus_coherence" in df_agent.columns:
        stim_column = "stimulus_coherence"
    else:
        raise ValueError("No stimulus column found in agent data")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Psychometric comparison
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_psychometric_comparison(ax1, df_agent, df_animal, metrics_agent, metrics_animal, stim_column)

    # Plot 2: Chronometric comparison
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_chronometric_comparison(ax2, df_agent, df_animal, metrics_agent, metrics_animal)

    # Plot 3: History effects
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_history_comparison(ax3, metrics_agent, metrics_animal)

    # Plot 4: Accuracy by coherence/contrast
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_accuracy_by_coherence(ax4, df_agent, df_animal)

    plt.suptitle(f"{title}\n{agent_name} vs {animal_name}", fontsize=16, fontweight="bold", y=0.98)

    # Encode figure
    figure_data_uri = _encode_figure(fig)

    # Create metrics table
    metrics_table = _create_metrics_table_html(metrics_agent, metrics_animal, per_session_stats)

    # Session info line
    session_info = f" | <strong>Sessions:</strong> {n_sessions}" if n_sessions > 1 else ""

    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8"/>
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
                background-color: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #1a1a2e;
                border-bottom: 3px solid {_COLOR_AGENT};
                padding-bottom: 0.5rem;
                margin-bottom: 1.5rem;
            }}
            h2 {{
                color: #555;
                margin-top: 2rem;
                margin-bottom: 1rem;
            }}
            .info-box {{
                background-color: #f0f0f0;
                border-left: 4px solid #888;
                padding: 1rem;
                margin: 1rem 0;
            }}
            .info-box p {{
                margin: 0.5rem 0;
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .metrics-table th {{
                background-color: #1a1a2e;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }}
            .metrics-table td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }}
            .metrics-table tr:hover {{
                background-color: #f5f5f5;
            }}
            .metrics-table tr.key-metric {{
                border-left: 4px solid {_COLOR_AGENT};
            }}
            .metrics-table tr.secondary-metric {{
                border-left: 4px solid #ccc;
            }}
            .match-cell {{
                font-weight: bold;
            }}
            .match-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 0.9em;
            }}
            .match-green {{
                background-color: #d4edda;
                color: #155724;
            }}
            .match-yellow {{
                background-color: #fff3cd;
                color: #856404;
            }}
            .match-red {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .sigma-legend {{
                margin-top: 0.5rem;
                font-size: 0.85rem;
                color: #666;
            }}
            .plot-section {{
                margin: 2rem 0;
                text-align: center;
            }}
            .plot-section img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin: 1.5rem 0;
            }}
            .stat-card {{
                background: white;
                border-left: 4px solid {_COLOR_AGENT};
                padding: 1.5rem;
                border-radius: 4px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
            .stat-card h3 {{
                margin: 0 0 0.5rem 0;
                font-size: 0.9rem;
                color: #666;
            }}
            .stat-card .value {{
                font-size: 2rem;
                font-weight: bold;
                margin: 0;
                color: #1a1a2e;
            }}
            pre {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 1rem;
                overflow-x: auto;
                font-size: 0.85rem;
            }}
            .expandable {{
                cursor: pointer;
                user-select: none;
            }}
            .expandable:before {{
                content: "\\25B6 ";
                display: inline-block;
                transition: transform 0.2s;
            }}
            .expandable.expanded:before {{
                transform: rotate(90deg);
            }}
            .collapsible-content {{
                display: none;
                padding-top: 1rem;
            }}
            .collapsible-content.show {{
                display: block;
            }}
        </style>
        <script>
            function toggleSection(id) {{
                const content = document.getElementById(id);
                const header = content.previousElementSibling;
                content.classList.toggle('show');
                header.classList.toggle('expanded');
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>

            <div class="info-box">
                <p><strong>Agent:</strong> {agent_name} ({agent_log_path.name})</p>
                <p><strong>Animal:</strong> {animal_name} ({animal_log_path.name}){session_info}</p>
                <p><strong>Task:</strong> {task_name}</p>
            </div>

            <div class="summary-stats">
                <div class="stat-card">
                    <h3>Agent Trials</h3>
                    <p class="value">{len(df_agent):,}</p>
                </div>
                <div class="stat-card">
                    <h3>Animal Trials</h3>
                    <p class="value">{len(df_animal):,}</p>
                </div>
                <div class="stat-card">
                    <h3>Agent Accuracy</h3>
                    <p class="value">{df_agent['correct'].mean():.1%}</p>
                </div>
                <div class="stat-card">
                    <h3>Animal Accuracy</h3>
                    <p class="value">{df_animal['correct'].mean():.1%}</p>
                </div>
            </div>

            <h2>Behavioral Comparison Plots</h2>
            <div class="plot-section">
                <img src="data:image/png;base64,{figure_data_uri}" alt="Behavioral Comparison"/>
            </div>

            <h2>Quantitative Metrics Comparison</h2>
            {metrics_table}

            <h2 class="expandable" onclick="toggleSection('agent-metrics')">Agent Metrics (JSON)</h2>
            <div id="agent-metrics" class="collapsible-content">
                <pre>{json.dumps(metrics_agent, indent=2)}</pre>
            </div>

            <h2 class="expandable" onclick="toggleSection('animal-metrics')">Animal Metrics (JSON)</h2>
            <div id="animal-metrics" class="collapsible-content">
                <pre>{json.dumps(metrics_animal, indent=2)}</pre>
            </div>
        </div>
    </body>
    </html>
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to: {output_path}")


__all__ = ["build_comparison_dashboard"]
