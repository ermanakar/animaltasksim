#!/usr/bin/env python3
"""Reroll saved PRL checkpoints with offline arbitration sidecar diagnostics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import mean
from typing import Sequence

import torch
import tyro

from agents.adaptive_control_config import AdaptiveControlConfig
from agents.adaptive_control_trainer import AdaptiveControlTrainer
from agents.losses import LossWeights


DIAGNOSTIC_FILENAME = "control_diagnostics.ndjson"
PER_RUN_SUMMARY_FILENAME = "per_run_reversal_window_summary.csv"
AGGREGATE_SUMMARY_FILENAME = "aggregate_reversal_window_summary.csv"
REVERSAL_WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("01-05", 1, 5),
    ("06-10", 6, 10),
    ("11-20", 11, 20),
    ("21-40", 21, 40),
    ("41-60", 41, 60),
    ("61-80", 61, 80),
)
SUMMARY_KEYS: tuple[str, ...] = (
    "correct",
    "reward",
    "base_stay_tendency",
    "raw_control_bias",
    "control_bias",
    "control_gate",
    "persistence_pressure",
    "exploration_pressure",
    "arbitration_adjustment",
    "raw_control_residual",
    "control_residual",
    "staleness_signal",
    "stay_tendency",
    "change_evidence",
    "history_retry_gate",
    "history_switch_gate",
)


@dataclass(slots=True)
class PRLArbitrationDiagnosticArgs:
    """Arguments for checkpoint rerolls with controller sidecar diagnostics."""

    source_root: Path = Path("runs/prl_adaptive_control_interaction_sweep_v1")
    output_root: Path = Path("runs/prl_arbitration_diagnostic_v1")
    conditions: Sequence[str] = (
        "exploration_only",
        "full_control_persist_half",
        "full_control_explore_double",
    )
    seeds: Sequence[int] = (42, 123, 456, 789, 2026)
    episodes: int | None = None
    trials_per_episode: int | None = None
    skip_existing: bool = True
    dry_run: bool = False

    def run(self) -> None:
        """Reroll selected checkpoints and summarize post-reversal traces."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        for condition in self.conditions:
            for seed in self.seeds:
                source_run = self.source_root / f"{condition}_seed{seed}"
                output_run = self.output_root / f"{condition}_seed{seed}"
                diagnostics_path = output_run / DIAGNOSTIC_FILENAME
                if self.dry_run:
                    print(f"[DRY RUN] {source_run} -> {diagnostics_path}")
                    continue
                if self.skip_existing and _reroll_is_complete(output_run, diagnostics_path):
                    print(f"[SKIP] {diagnostics_path}")
                    continue
                print(f"[REROLL] {source_run} -> {output_run}")
                _reroll_checkpoint(
                    source_run=source_run,
                    output_run=output_run,
                    diagnostics_path=diagnostics_path,
                    episodes=self.episodes,
                    trials_per_episode=self.trials_per_episode,
                )

        if self.dry_run:
            return

        per_run_rows = self._collect_per_run_rows()
        aggregate_rows = _aggregate_rows(per_run_rows)
        _write_csv(self.output_root / PER_RUN_SUMMARY_FILENAME, per_run_rows)
        _write_csv(self.output_root / AGGREGATE_SUMMARY_FILENAME, aggregate_rows)
        _print_summary(aggregate_rows)

    def _collect_per_run_rows(self) -> list[dict[str, object]]:
        """Summarize sidecars from every requested condition and seed."""
        rows: list[dict[str, object]] = []
        for condition in self.conditions:
            for seed in self.seeds:
                path = self.output_root / f"{condition}_seed{seed}" / DIAGNOSTIC_FILENAME
                if not path.exists():
                    print(f"[MISSING] {path}")
                    continue
                records = _load_records(path)
                rows.extend(_summarize_reversal_windows(records, condition=condition, seed=seed))
        return rows


def _reroll_checkpoint(
    *,
    source_run: Path,
    output_run: Path,
    diagnostics_path: Path,
    episodes: int | None,
    trials_per_episode: int | None,
) -> None:
    """Load one checkpoint and reroll it with a non-contract sidecar log."""
    config = _load_source_config(
        source_run / "config.json",
        output_run=output_run,
        episodes=episodes,
        trials_per_episode=trials_per_episode,
    )
    model_path = source_run / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {model_path}")

    output_run.mkdir(parents=True, exist_ok=True)
    # Clear the stale manifest first so an interrupted reroll cannot leave a
    # manifest that falsely marks the (now partial) run as complete.
    for generated_path in (
        output_run / "trials.ndjson",
        diagnostics_path,
        output_run / "diagnostic_manifest.json",
    ):
        generated_path.unlink(missing_ok=True)

    trainer = AdaptiveControlTrainer(config)
    trainer.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    paths = config.output_paths()
    rollout_stats = trainer.rollout(paths, diagnostics_path=diagnostics_path)
    trainer.save(paths, training_metrics={}, rollout_stats=rollout_stats)
    diagnostic_rows = len(_load_records(diagnostics_path)) if diagnostics_path.exists() else 0
    manifest = {
        "source_run": str(source_run),
        "source_checkpoint": str(model_path),
        "diagnostics": str(diagnostics_path),
        "diagnostic_rows": diagnostic_rows,
        "rollout_stats": rollout_stats,
    }
    # Written last, so its presence + matching row count marks a completed reroll.
    (output_run / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _load_source_config(
    path: Path,
    *,
    output_run: Path,
    episodes: int | None,
    trials_per_episode: int | None,
) -> AdaptiveControlConfig:
    """Rebuild an adaptive-control config from a saved source run."""
    if not path.exists():
        raise FileNotFoundError(f"Missing source config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(AdaptiveControlConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    for key in ("reference_log", "output_dir"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    if isinstance(values.get("loss_weights"), dict):
        values["loss_weights"] = LossWeights(**values["loss_weights"])
    values["output_dir"] = output_run
    values["epochs"] = 0
    if episodes is not None:
        values["episodes"] = episodes
    if trials_per_episode is not None:
        values["trials_per_episode"] = trials_per_episode
    config = AdaptiveControlConfig(**values)
    if config.task != "prl":
        raise ValueError(f"PRL arbitration diagnostic requires task='prl', got {config.task!r}")
    return config


def _reroll_is_complete(output_run: Path, diagnostics_path: Path) -> bool:
    """Skip only a verifiably complete reroll, never an interrupted partial one.

    The manifest is written last and records the expected sidecar row count, so a
    truncated sidecar (e.g. an interrupted run) is detected and re-run.
    """
    manifest_path = output_run / "diagnostic_manifest.json"
    if not manifest_path.exists() or not diagnostics_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    expected = manifest.get("diagnostic_rows")
    if not isinstance(expected, int) or expected <= 0:
        return False
    try:
        actual_rows = len(_load_records(diagnostics_path))
    except (json.JSONDecodeError, OSError):
        # A malformed/truncated sidecar is treated as incomplete, not fatal.
        return False
    return actual_rows == expected


def _load_records(path: Path) -> list[dict[str, object]]:
    """Load compact sidecar records from one diagnostic reroll."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summarize_reversal_windows(
    records: Sequence[dict[str, object]],
    *,
    condition: str,
    seed: int,
) -> list[dict[str, object]]:
    """Aggregate controller traces into post-reversal windows."""
    window_records: dict[str, list[dict[str, object]]] = {
        label: [] for label, _, _ in REVERSAL_WINDOWS
    }
    offset_by_session: dict[str, int | None] = {}
    for record in records:
        session_id = str(record["session_id"])
        offset = offset_by_session.get(session_id)
        if bool(record.get("reversal", False)):
            offset = 1
        elif offset is not None:
            offset += 1
        offset_by_session[session_id] = offset
        if offset is None:
            continue
        label = _window_label(offset)
        if label is not None:
            window_records[label].append(record)

    rows: list[dict[str, object]] = []
    for label, _, _ in REVERSAL_WINDOWS:
        selected = window_records[label]
        row: dict[str, object] = {
            "condition": condition,
            "seed": seed,
            "window": label,
            "trial_count": len(selected),
        }
        for key in SUMMARY_KEYS:
            row[f"{key}_mean"] = _mean_numeric(selected, key)
        rows.append(row)
    return rows


def _window_label(offset: int) -> str | None:
    """Return the configured label containing one post-reversal offset."""
    for label, start, end in REVERSAL_WINDOWS:
        if start <= offset <= end:
            return label
    return None


def _mean_numeric(records: Sequence[dict[str, object]], key: str) -> float | None:
    """Return a finite numeric mean when a diagnostic key is available."""
    values = [
        float(record[key])
        for record in records
        if isinstance(record.get(key), (int, float, bool))
    ]
    return mean(values) if values else None


def _aggregate_rows(per_run_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-seed window summaries into condition means."""
    rows: list[dict[str, object]] = []
    conditions = sorted({str(row["condition"]) for row in per_run_rows})
    for condition in conditions:
        for label, _, _ in REVERSAL_WINDOWS:
            selected = [
                row
                for row in per_run_rows
                if row["condition"] == condition and row["window"] == label
            ]
            row: dict[str, object] = {
                "condition": condition,
                "window": label,
                "seed_count": len(selected),
                "trial_count": sum(int(item["trial_count"]) for item in selected),
            }
            for key in SUMMARY_KEYS:
                row[f"{key}_mean"] = _mean_numeric(selected, f"{key}_mean")
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write rows to CSV with stable field order."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: Sequence[dict[str, object]]) -> None:
    """Print the compact controller decomposition scorecard."""
    print("\n" + "=" * 104)
    print("PRL ARBITRATION REVERSAL-WINDOW DIAGNOSTIC")
    print("=" * 104)
    print(
        f"{'condition':<32} | {'window':>6} | {'n':>5} | {'optimal':>7} | "
        f"{'persist':>8} | {'explore':>8} | {'chg_evid':>8} | {'retry_g':>8} | "
        f"{'switch_g':>8} | {'resid':>8}"
    )
    print("-" * 124)
    for row in rows:
        print(
            f"{str(row['condition']):<32} | {str(row['window']):>6} | "
            f"{int(row['trial_count']):>5} | {_fmt(row.get('correct_mean')):>7} | "
            f"{_fmt(row.get('persistence_pressure_mean')):>8} | "
            f"{_fmt(row.get('exploration_pressure_mean')):>8} | "
            f"{_fmt(row.get('change_evidence_mean')):>8} | "
            f"{_fmt(row.get('history_retry_gate_mean')):>8} | "
            f"{_fmt(row.get('history_switch_gate_mean')):>8} | "
            f"{_fmt(row.get('control_residual_mean')):>8}"
        )


def _fmt(value: object) -> str:
    """Format one optional scorecard value."""
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.3f}"


def main() -> None:
    """CLI entry point."""
    tyro.cli(PRLArbitrationDiagnosticArgs).run()


__all__ = [
    "AGGREGATE_SUMMARY_FILENAME",
    "DIAGNOSTIC_FILENAME",
    "PER_RUN_SUMMARY_FILENAME",
    "PRLArbitrationDiagnosticArgs",
    "REVERSAL_WINDOWS",
    "SUMMARY_KEYS",
    "main",
]


if __name__ == "__main__":
    main()
