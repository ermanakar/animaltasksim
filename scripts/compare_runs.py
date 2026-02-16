"""Batch comparison of all agent runs against animal reference data.

Scans the runs/ directory for metrics.json files (computing them from
trials.ndjson when missing), loads the cached reference metrics, and
produces a ranked HTML leaderboard showing how each agent compares to
animal behavioral fingerprints.

Usage:
    python scripts/compare_runs.py                          # all runs, auto-detect task
    python scripts/compare_runs.py --task ibl_2afc          # only IBL runs
    python scripts/compare_runs.py --task rdm               # only RDM runs
    python scripts/compare_runs.py --recompute              # force recompute metrics
    python scripts/compare_runs.py --out comparison.html    # custom output path
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from animaltasksim.config import ProjectPaths
from eval.metrics import load_and_compute


# ------------------------------------------------------------------ #
# Reference data loading
# ------------------------------------------------------------------ #

_REFERENCE_MAP: dict[str, dict[str, str]] = {
    "ibl_2afc": {
        "log": "data/ibl/reference.ndjson",
        "metrics": "data/ibl/reference_metrics.json",
        "label": "IBL Mouse",
    },
    "rdm": {
        "log": "data/macaque/reference.ndjson",
        "metrics": "data/macaque/reference_metrics.json",
        "label": "Macaque (Roitman & Shadlen)",
    },
}


def _load_reference(task: str, root: Path) -> dict:
    """Load cached reference metrics, computing if absent."""
    info = _REFERENCE_MAP[task]
    metrics_path = root / info["metrics"]
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    log_path = root / info["log"]
    if not log_path.exists():
        raise FileNotFoundError(f"Reference log not found: {log_path}")
    metrics = load_and_compute(log_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ------------------------------------------------------------------ #
# Run discovery and metrics loading
# ------------------------------------------------------------------ #


def _detect_task(metrics: dict) -> str | None:
    """Infer task from metrics structure."""
    chrono = metrics.get("chronometric", {})
    unit = chrono.get("slope_unit", "")
    if "contrast" in unit:
        return "ibl_2afc"
    if "coherence" in unit:
        return "rdm"
    return None


def _scan_runs(
    runs_dir: Path, task_filter: str | None, recompute: bool
) -> list[dict]:
    """Discover runs and load/compute metrics for each."""
    results = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        log_path = run_dir / "trials.ndjson"

        # Load or compute metrics
        metrics = None
        if metrics_path.exists() and not recompute:
            try:
                raw = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics = raw.get("metrics", raw)
            except (json.JSONDecodeError, KeyError):
                pass

        if metrics is None and log_path.exists():
            try:
                metrics = load_and_compute(log_path)
                if metrics:
                    payload = {"log": str(log_path), "metrics": metrics}
                    metrics_path.write_text(
                        json.dumps(payload, indent=2), encoding="utf-8"
                    )
            except Exception:
                continue

        if not metrics:
            continue

        task = _detect_task(metrics)
        if task is None:
            continue
        if task_filter and task != task_filter:
            continue

        results.append({
            "name": run_dir.name,
            "task": task,
            "metrics": metrics,
            "path": str(run_dir),
        })
    return results


# ------------------------------------------------------------------ #
# Metric extraction and scoring
# ------------------------------------------------------------------ #

_KEY_METRICS = [
    ("psychometric", "slope", "Psychometric slope"),
    ("psychometric", "bias", "Bias"),
    ("chronometric", "slope_ms_per_unit", "Chrono slope (ms/unit)"),
    ("chronometric", "intercept_ms", "RT intercept (ms)"),
    ("chronometric", "ceiling_fraction", "RT ceiling frac"),
    ("history", "win_stay", "Win-stay"),
    ("history", "lose_shift", "Lose-shift"),
    ("history", "sticky_choice", "Sticky choice"),
]


def _get(metrics: dict, group: str, key: str) -> float | None:
    """Safely extract a metric value."""
    sub = metrics.get(group)
    if not isinstance(sub, dict):
        return None
    val = sub.get(key)
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _pct_match(agent_val: float | None, ref_val: float | None) -> float | None:
    """Percentage deviation from reference (lower = better)."""
    if agent_val is None or ref_val is None:
        return None
    if ref_val == 0:
        return 0.0 if agent_val == 0 else 100.0
    return abs(agent_val - ref_val) / abs(ref_val) * 100


def _composite_score(agent: dict, ref: dict) -> float:
    """Compute a composite match score (0 = perfect match, 100 = complete miss).

    Weights: psychometric slope 25%, chrono slope 25%, bias 15%,
    win-stay 15%, lose-shift 10%, sticky 10%.
    """
    weights = {
        ("psychometric", "slope"): 0.25,
        ("chronometric", "slope_ms_per_unit"): 0.25,
        ("psychometric", "bias"): 0.15,
        ("history", "win_stay"): 0.15,
        ("history", "lose_shift"): 0.10,
        ("history", "sticky_choice"): 0.10,
    }
    total_weight = 0.0
    score = 0.0
    for (group, key), w in weights.items():
        pct = _pct_match(_get(agent, group, key), _get(ref, group, key))
        if pct is not None:
            score += w * min(pct, 200.0)  # cap at 200% to avoid outlier domination
            total_weight += w
    return score / total_weight if total_weight > 0 else 100.0


# ------------------------------------------------------------------ #
# HTML rendering
# ------------------------------------------------------------------ #


def _color_class(pct: float | None) -> str:
    """CSS class from deviation percentage."""
    if pct is None:
        return "na"
    if pct <= 10:
        return "good"
    if pct <= 25:
        return "ok"
    if pct <= 50:
        return "warn"
    return "bad"


def _fmt(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _render_html(
    runs: list[dict],
    ref_by_task: dict[str, dict],
    output_path: Path,
) -> None:
    """Render a comparison leaderboard as self-contained HTML."""

    # Group runs by task
    by_task: dict[str, list[dict]] = {}
    for run in runs:
        by_task.setdefault(run["task"], []).append(run)

    sections = []
    for task in ["ibl_2afc", "rdm"]:
        task_runs = by_task.get(task, [])
        if not task_runs:
            continue
        ref = ref_by_task.get(task)
        if not ref:
            continue

        ref_label = _REFERENCE_MAP[task]["label"]

        # Score and sort runs
        for r in task_runs:
            r["_score"] = _composite_score(r["metrics"], ref)
        task_runs.sort(key=lambda r: r["_score"])

        # Build table rows
        rows = []
        # Reference row
        ref_cells = "".join(
            f'<td class="ref">{_fmt(_get(ref, g, k))}</td>'
            for g, k, _ in _KEY_METRICS
        )
        rows.append(
            f'<tr class="ref-row"><td>🏆</td><td><b>{ref_label}</b></td>'
            f"{ref_cells}"
            f'<td class="ref">REF</td></tr>'
        )

        # Agent rows
        for rank, r in enumerate(task_runs, 1):
            m = r["metrics"]
            quality = m.get("quality", {})
            row_class = "degenerate" if quality.get("degenerate") else ""
            ceiling_flag = " ⚠️" if quality.get("rt_ceiling_warning") else ""

            cells = []
            for g, k, _ in _KEY_METRICS:
                val = _get(m, g, k)
                pct = _pct_match(val, _get(ref, g, k))
                cls = _color_class(pct)
                pct_str = f' <small>({_fmt(pct, 0)}%)</small>' if pct is not None else ""
                cells.append(f'<td class="{cls}">{_fmt(val)}{pct_str}</td>')

            score = r["_score"]
            score_cls = _color_class(score)
            rows.append(
                f'<tr class="{row_class}">'
                f"<td>{rank}</td>"
                f"<td>{r['name']}{ceiling_flag}</td>"
                f"{''.join(cells)}"
                f'<td class="{score_cls}"><b>{_fmt(score, 1)}%</b></td>'
                f"</tr>"
            )

        # Header row
        header = "<th>#</th><th>Run</th>"
        header += "".join(f"<th>{label}</th>" for _, _, label in _KEY_METRICS)
        header += "<th>Score</th>"

        task_label = "Mouse IBL 2AFC" if task == "ibl_2afc" else "Macaque RDM"
        sections.append(f"""
        <h2>{task_label}</h2>
        <p>Reference: <b>{ref_label}</b> · {len(task_runs)} agent runs · Sorted by composite match score (lower = closer to animal)</p>
        <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AnimalTaskSim — Agent vs Animal Leaderboard</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
    h2 {{ color: #16213e; margin-top: 40px; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
             box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin: 16px 0 32px; font-size: 13px; }}
    th {{ background: #16213e; color: white; padding: 10px 8px;
          text-align: left; font-weight: 600; white-space: nowrap; }}
    td {{ padding: 8px; border-bottom: 1px solid #e9ecef; }}
    tr:hover {{ background: #f1f3f5; }}
    .ref-row {{ background: #e8f5e9 !important; font-weight: 500; }}
    .ref-row:hover {{ background: #c8e6c9 !important; }}
    .ref {{ color: #2e7d32; }}
    .good {{ color: #2e7d32; background: #e8f5e9; }}
    .ok {{ color: #f57f17; background: #fff8e1; }}
    .warn {{ color: #e65100; background: #fff3e0; }}
    .bad {{ color: #b71c1c; background: #ffebee; }}
    .na {{ color: #9e9e9e; }}
    .degenerate td {{ opacity: 0.6; }}
    small {{ color: #757575; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; font-size: 13px; }}
    .legend span {{ padding: 2px 10px; border-radius: 4px; }}
    .meta {{ color: #616161; font-size: 13px; margin-top: 8px; }}
    .score-info {{ background: #f5f5f5; padding: 12px 16px; border-radius: 6px;
                   margin: 8px 0 20px; font-size: 13px; color: #424242; }}
</style>
</head>
<body>
<h1>AnimalTaskSim — Agent vs Animal Leaderboard</h1>
<p class="meta">Generated by <code>scripts/compare_runs.py</code> · Composite score weights:
psychometric slope 25%, chronometric slope 25%, bias 15%, win-stay 15%, lose-shift 10%, sticky 10%</p>

<div class="legend">
    <span class="good">≤10% deviation</span>
    <span class="ok">10–25%</span>
    <span class="warn">25–50%</span>
    <span class="bad">&gt;50%</span>
    <span class="na">No data</span>
    <span>⚠️ = RT ceiling warning</span>
</div>

<div class="score-info">
    <b>Score</b>: Weighted average of per-metric percentage deviations from the animal reference.
    0% = perfect match. Degenerate runs (extreme bias, collapsed history, or flat chronometry) are dimmed.
    Percentage in parentheses shows per-metric deviation from reference.
</div>

{"".join(sections)}

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #


@dataclass(slots=True)
class CompareArgs:
    """Compare all agent runs against animal reference data."""

    task: Literal["ibl_2afc", "rdm"] | None = None
    """Filter to a specific task. If None, include both."""

    out: Path = Path("runs/leaderboard.html")
    """Output path for the leaderboard HTML."""

    recompute: bool = False
    """Force recomputation of metrics for all runs."""


def main(args: CompareArgs) -> None:
    paths = ProjectPaths.from_cwd()

    # Load references
    ref_by_task: dict[str, dict] = {}
    tasks = [args.task] if args.task else ["ibl_2afc", "rdm"]
    for task in tasks:
        try:
            ref_by_task[task] = _load_reference(task, paths.root)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    # Scan runs
    runs = _scan_runs(paths.runs, args.task, args.recompute)
    print(f"Found {len(runs)} runs with metrics")

    # Render
    _render_html(runs, ref_by_task, args.out)
    print(f"Leaderboard written to {args.out}")


if __name__ == "__main__":
    main(tyro.cli(CompareArgs))
