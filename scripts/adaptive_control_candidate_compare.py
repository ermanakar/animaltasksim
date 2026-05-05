#!/usr/bin/env python3
"""Compare selected adaptive-control settings across multiple seeds."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import tyro


def _slug(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(slots=True)
class CandidateCompareArgs:
    """Arguments for adaptive-control multi-seed candidate comparison."""

    run_root: Path = Path("runs/adaptive_control_candidate_compare")
    seeds: Sequence[int] = (42, 123, 456)
    persistence_bias_scales: Sequence[float] = (1.6,)
    persistence_learning_rates: Sequence[float] = (0.8,)
    include_no_persistence_control: bool = False
    task: Literal["ibl_2afc", "rdm"] = "ibl_2afc"
    episodes: int = 20
    trials_per_episode: int = 400
    epochs: int = 5
    hidden_size: int = 64
    learning_rate: float = 1e-3
    max_sessions: int = 80
    max_trials_per_session: int = 128
    min_commit_steps: int = 5
    max_commit_steps: int = 300
    drift_scale: float = 6.0
    history_bias_scale: float = 2.0
    history_drift_scale: float = 0.3
    lapse_rate: float = 0.05
    control_uncertainty_power: float = 2.0
    dry_run: bool = False

    def run(self) -> None:
        """Execute the comparison and write summaries."""
        if len(self.persistence_bias_scales) != len(self.persistence_learning_rates):
            raise ValueError("persistence_bias_scales and persistence_learning_rates must have the same length")

        self.run_root.mkdir(parents=True, exist_ok=True)
        per_run_rows: list[dict[str, object]] = []

        candidates: list[tuple[str, bool, float, float]] = []
        if self.include_no_persistence_control:
            candidates.append(("no_persistence", False, 0.0, 0.0))
        for bias_scale, learning_rate in zip(self.persistence_bias_scales, self.persistence_learning_rates, strict=True):
            label = f"pb{_slug(bias_scale)}_plr{_slug(learning_rate)}"
            candidates.append((label, True, bias_scale, learning_rate))

        for label, persistence_enabled, bias_scale, learning_rate in candidates:
            for seed in self.seeds:
                run_dir = self.run_root / f"{label}_seed{seed}"
                print(f"\n{'=' * 80}")
                print(f"Running candidate {label} | seed={seed}")
                print(f"Output: {run_dir}")
                print(f"{'=' * 80}")
                _run_command(
                    self._build_train_command(
                        run_dir=run_dir,
                        seed=seed,
                        persistence_enabled=persistence_enabled,
                        persistence_bias_scale=bias_scale,
                        persistence_learning_rate=learning_rate,
                    ),
                    dry_run=self.dry_run,
                )
                _run_command(self._build_eval_command(run_dir), dry_run=self.dry_run)
                if not self.dry_run:
                    row = self._summarize_run(run_dir)
                    row.update(
                        {
                            "label": label,
                            "seed": seed,
                            "persistence_enabled": persistence_enabled,
                            "persistence_bias_scale": bias_scale if persistence_enabled else None,
                            "persistence_learning_rate": learning_rate if persistence_enabled else None,
                        }
                    )
                    per_run_rows.append(row)

        if self.dry_run or not per_run_rows:
            return

        per_run_rows.sort(key=lambda row: (str(row.get("label")), int(row.get("seed", 0))))
        per_run_csv = self.run_root / "per_run_summary.csv"
        per_run_fields = sorted({key for row in per_run_rows for key in row.keys()})
        with per_run_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_run_fields)
            writer.writeheader()
            for row in per_run_rows:
                writer.writerow(row)

        grouped: dict[str, list[dict[str, object]]] = {}
        for row in per_run_rows:
            grouped.setdefault(str(row["label"]), []).append(row)

        aggregate_rows: list[dict[str, object]] = []
        for label, rows in grouped.items():
            aggregate = {
                "label": label,
                "num_seeds": len(rows),
                "degenerate_count": sum(1 for row in rows if row.get("degenerate") is True),
                "chronometric_ok_count": sum(1 for row in rows if row.get("chronometric_ok") is True),
            }
            template = rows[0]
            aggregate["persistence_enabled"] = template.get("persistence_enabled")
            aggregate["persistence_bias_scale"] = template.get("persistence_bias_scale")
            aggregate["persistence_learning_rate"] = template.get("persistence_learning_rate")

            metric_keys = [
                "psychometric_slope",
                "chronometric_slope",
                "win_stay",
                "lose_shift",
                "sticky_choice",
                "retry_after_failure_weak",
                "retry_after_failure_strong",
                "switch_after_failure_weak",
                "switch_after_failure_strong",
                "switch_gap",
                "switch_after_streak_weak",
                "switch_after_streak_strong",
                "exploration_gap",
                "retry_gap",
            ]
            for key in metric_keys:
                values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
                if values:
                    aggregate[f"{key}_mean"] = _mean(values)
                    aggregate[f"{key}_std"] = _std(values) if len(values) > 1 else 0.0
            aggregate_rows.append(aggregate)

        aggregate_rows.sort(
            key=lambda row: (
                float(row.get("retry_gap_mean") or float("-inf")),
                float(row.get("psychometric_slope_mean") or float("-inf")),
            ),
            reverse=True,
        )

        aggregate_csv = self.run_root / "candidate_summary.csv"
        aggregate_fields = sorted({key for row in aggregate_rows for key in row.keys()})
        with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
            writer.writeheader()
            for row in aggregate_rows:
                writer.writerow(row)

        aggregate_json = self.run_root / "candidate_summary.json"
        aggregate_json.write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")

        print(f"\n{'=' * 80}")
        print("Adaptive Control Candidate Comparison")
        print(f"{'=' * 80}")
        print(
            f"{'label':<18} | {'retry':>7} | {'switch':>7} | {'explore':>7} | "
            f"{'psych':>7} | {'chrono':>8} | {'deg':>3}"
        )
        print("-" * 80)
        for row in aggregate_rows:
            retry_gap = row.get("retry_gap_mean")
            switch_gap = row.get("switch_gap_mean")
            exploration_gap = row.get("exploration_gap_mean")
            psych = row.get("psychometric_slope_mean")
            chrono = row.get("chronometric_slope_mean")
            retry_gap_str = f"{retry_gap:.3f}" if isinstance(retry_gap, (int, float)) else "n/a"
            switch_gap_str = f"{switch_gap:.3f}" if isinstance(switch_gap, (int, float)) else "n/a"
            exploration_gap_str = f"{exploration_gap:.3f}" if isinstance(exploration_gap, (int, float)) else "n/a"
            psych_str = f"{psych:.2f}" if isinstance(psych, (int, float)) else "n/a"
            chrono_str = f"{chrono:.2f}" if isinstance(chrono, (int, float)) else "n/a"
            print(
                f"{str(row['label']):<18} | {retry_gap_str:>7} | "
                f"{switch_gap_str:>7} | {exploration_gap_str:>7} | "
                f"{psych_str:>7} | {chrono_str:>8} | {int(row['degenerate_count']):>3}"
            )

        print(f"\nPer-run summary saved to {per_run_csv}")
        print(f"Candidate summary saved to {aggregate_csv}")

    def _build_train_command(
        self,
        *,
        run_dir: Path,
        seed: int,
        persistence_enabled: bool,
        persistence_bias_scale: float,
        persistence_learning_rate: float,
    ) -> list[str]:
        cmd = [
            sys.executable,
            "scripts/train_adaptive_control.py",
            "--output-dir",
            str(run_dir),
            "--task",
            self.task,
            "--seed",
            str(seed),
            "--episodes",
            str(self.episodes),
            "--trials-per-episode",
            str(self.trials_per_episode),
            "--epochs",
            str(self.epochs),
            "--hidden-size",
            str(self.hidden_size),
            "--learning-rate",
            str(self.learning_rate),
            "--control-profile",
            "persistence_only" if persistence_enabled else "no_control",
            "--max-sessions",
            str(self.max_sessions),
            "--max-trials-per-session",
            str(self.max_trials_per_session),
            "--min-commit-steps",
            str(self.min_commit_steps),
            "--max-commit-steps",
            str(self.max_commit_steps),
            "--drift-scale",
            str(self.drift_scale),
            "--history-bias-scale",
            str(self.history_bias_scale),
            "--history-drift-scale",
            str(self.history_drift_scale),
            "--lapse-rate",
            str(self.lapse_rate),
            "--persistence-bias-scale",
            str(persistence_bias_scale),
            "--persistence-learning-rate",
            str(persistence_learning_rate),
            "--control-uncertainty-power",
            str(self.control_uncertainty_power),
        ]
        return cmd

    @staticmethod
    def _build_eval_command(run_dir: Path) -> list[str]:
        return [sys.executable, "scripts/evaluate_agent.py", "--run", str(run_dir)]

    @staticmethod
    def _summarize_run(run_dir: Path) -> dict[str, object]:
        payload = _load_json(run_dir / "metrics.json")
        if payload is None:
            return {"run_dir": str(run_dir)}
        metrics = payload.get("metrics", payload)
        psychometric = metrics.get("psychometric", {})
        chronometric = metrics.get("chronometric", {})
        history = metrics.get("history", {})
        probe = metrics.get("adaptive_control_probe", {})
        exploration_probe = metrics.get("exploration_probe", {})
        quality = metrics.get("quality", {})
        row: dict[str, object] = {
            "run_dir": str(run_dir),
            "psychometric_slope": psychometric.get("slope"),
            "chronometric_slope": chronometric.get("slope_ms_per_unit"),
            "win_stay": history.get("win_stay"),
            "lose_shift": history.get("lose_shift"),
            "sticky_choice": history.get("sticky_choice"),
            "retry_after_failure_weak": probe.get("retry_after_failure_weak"),
            "retry_after_failure_strong": probe.get("retry_after_failure_strong"),
            "switch_after_failure_weak": probe.get("switch_after_failure_weak"),
            "switch_after_failure_strong": probe.get("switch_after_failure_strong"),
            "weak_failure_count": probe.get("weak_failure_count"),
            "strong_failure_count": probe.get("strong_failure_count"),
            "switch_after_streak_weak": exploration_probe.get("switch_after_streak_weak"),
            "switch_after_streak_strong": exploration_probe.get("switch_after_streak_strong"),
            "weak_streak_count": exploration_probe.get("weak_streak_count"),
            "strong_streak_count": exploration_probe.get("strong_streak_count"),
            "chronometric_ok": quality.get("chronometric_ok"),
            "degenerate": quality.get("degenerate"),
        }
        weak_retry = row.get("retry_after_failure_weak")
        strong_retry = row.get("retry_after_failure_strong")
        if isinstance(weak_retry, (int, float)) and isinstance(strong_retry, (int, float)):
            row["retry_gap"] = float(weak_retry) - float(strong_retry)
        weak_switch = row.get("switch_after_failure_weak")
        strong_switch = row.get("switch_after_failure_strong")
        if isinstance(weak_switch, (int, float)) and isinstance(strong_switch, (int, float)):
            row["switch_gap"] = float(strong_switch) - float(weak_switch)
        weak_explore = row.get("switch_after_streak_weak")
        strong_explore = row.get("switch_after_streak_strong")
        if isinstance(weak_explore, (int, float)) and isinstance(strong_explore, (int, float)):
            row["exploration_gap"] = float(weak_explore) - float(strong_explore)
        return row


def main() -> None:
    """CLI entry point."""
    tyro.cli(CandidateCompareArgs).run()


if __name__ == "__main__":
    main()
