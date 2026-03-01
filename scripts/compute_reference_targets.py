"""Compute per-session behavioral targets from reference data.

Loads a reference .ndjson file, computes psychometric, chronometric, and
history metrics for each session independently, then reports mean +/- std
and aggregate statistics.  Outputs a structured JSON file suitable for use
as the single source of truth for behavioral targets.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from eval.metrics import (
    compute_all_metrics,
    load_trials,
)


@dataclass(slots=True)
class Args:
    """Compute per-session reference targets."""

    reference_path: str = "data/ibl/reference.ndjson"
    task: str = "ibl_2afc"
    output: str | None = None  # default: same directory as reference, named reference_targets.json


def _extract_scalar(metrics: dict, *keys: str) -> float | None:
    """Walk nested dict to extract a scalar value."""
    obj: object = metrics
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    if obj is None or (isinstance(obj, float) and not np.isfinite(obj)):
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    return None


def _stats(values: list[float]) -> dict[str, float | int]:
    """Compute mean, std, median, min, max for a list of finite floats."""
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }


def main() -> None:
    args = tyro.cli(Args)
    ref_path = Path(args.reference_path)
    if not ref_path.exists():
        print(f"Error: {ref_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load all trials
    df = load_trials(ref_path)
    if df.empty:
        print("Error: no trials loaded", file=sys.stderr)
        sys.exit(1)

    task = args.task
    session_ids = sorted(df["session_id"].unique())
    print(f"Source: {ref_path} ({len(session_ids)} sessions, {len(df)} trials)")
    print()

    # --- Per-session analysis ---
    per_session: list[dict] = []
    psych_slopes: list[float] = []
    chrono_slopes: list[float] = []
    win_stays: list[float] = []
    lose_shifts: list[float] = []
    lapse_lows: list[float] = []
    lapse_highs: list[float] = []
    degenerate_sessions: list[str] = []

    for sid in session_ids:
        session_df = df[df["session_id"] == sid].copy()
        n_trials = len(session_df)
        m = compute_all_metrics(session_df, task)

        ps = _extract_scalar(m, "psychometric", "slope")
        cs = _extract_scalar(m, "chronometric", "slope_ms_per_unit")
        ws = _extract_scalar(m, "history", "win_stay")
        ls = _extract_scalar(m, "history", "lose_shift")
        ll = _extract_scalar(m, "psychometric", "lapse_low")
        lh = _extract_scalar(m, "psychometric", "lapse_high")

        is_degenerate = ps is not None and ps > 100.0
        entry = {
            "session_id": sid,
            "n_trials": n_trials,
            "psych_slope": ps,
            "chrono_slope_ms_per_unit": cs,
            "win_stay": ws,
            "lose_shift": ls,
            "lapse_low": ll,
            "lapse_high": lh,
            "degenerate": is_degenerate,
        }
        per_session.append(entry)

        # Print per-session row
        flag = " *** DEGENERATE" if is_degenerate else ""
        print(
            f"  {sid[:8]}...  n={n_trials:4d}  "
            f"psych={ps:7.1f}  chrono={cs if cs is not None else float('nan'):7.1f}  "
            f"WS={ws if ws is not None else float('nan'):.3f}  LS={ls if ls is not None else float('nan'):.3f}"
            f"{flag}"
        )

        # Collect for stats (excluding degenerate psych fits)
        if not is_degenerate and ps is not None:
            psych_slopes.append(ps)
        if cs is not None:
            chrono_slopes.append(cs)
        if ws is not None:
            win_stays.append(ws)
        if ls is not None:
            lose_shifts.append(ls)
        if ll is not None:
            lapse_lows.append(ll)
        if lh is not None:
            lapse_highs.append(lh)
        if is_degenerate:
            degenerate_sessions.append(sid)

    print()
    n_degen = len(degenerate_sessions)
    if n_degen:
        print(f"Degenerate sessions excluded from psych stats: {n_degen}")
    print()

    # --- Per-session statistics ---
    stats = {
        "psych_slope": _stats(psych_slopes) if psych_slopes else None,
        "chrono_slope_ms_per_unit": _stats(chrono_slopes) if chrono_slopes else None,
        "win_stay": _stats(win_stays) if win_stays else None,
        "lose_shift": _stats(lose_shifts) if lose_shifts else None,
        "lapse_low": _stats(lapse_lows) if lapse_lows else None,
        "lapse_high": _stats(lapse_highs) if lapse_highs else None,
    }

    print("Per-session targets (mean +/- std):")
    for name, s in stats.items():
        if s is None:
            print(f"  {name:30s}  no data")
        else:
            print(f"  {name:30s}  {s['mean']:8.2f} +/- {s['std']:6.2f}  (median {s['median']:8.2f}, range [{s['min']:.1f}, {s['max']:.1f}], n={s['n']})")
    print()

    # --- Aggregate (all trials pooled) ---
    aggregate = compute_all_metrics(df, task)
    agg_summary = {
        "psych_slope": _extract_scalar(aggregate, "psychometric", "slope"),
        "chrono_slope_ms_per_unit": _extract_scalar(aggregate, "chronometric", "slope_ms_per_unit"),
        "win_stay": _extract_scalar(aggregate, "history", "win_stay"),
        "lose_shift": _extract_scalar(aggregate, "history", "lose_shift"),
        "lapse_low": _extract_scalar(aggregate, "psychometric", "lapse_low"),
        "lapse_high": _extract_scalar(aggregate, "psychometric", "lapse_high"),
    }

    print("Aggregate (all trials pooled):")
    for name, val in agg_summary.items():
        print(f"  {name:30s}  {val if val is not None else 'N/A'}")
    print()

    # --- Contrast levels ---
    if task == "ibl_2afc":
        stim_key = "stimulus_contrast"
    else:
        stim_key = "stimulus_coherence"
    if stim_key in df.columns:
        levels = sorted(df[stim_key].abs().unique())
        print(f"Stimulus levels (|{stim_key}|): {levels}")
    print()

    # --- Output JSON ---
    output_data = {
        "source": str(ref_path),
        "task": task,
        "n_sessions": len(session_ids),
        "n_trials": len(df),
        "degenerate_sessions": degenerate_sessions,
        "per_session": per_session,
        "per_session_stats": stats,
        "aggregate": agg_summary,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = ref_path.parent / "reference_targets.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
