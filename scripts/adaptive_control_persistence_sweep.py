#!/usr/bin/env python3
"""Single-seed sweep for adaptive-control persistence calibration.

Runs a matched set of adaptive-control experiments on IBL 2AFC to quantify how
the persistence controller changes rollout behavior, especially the
retry-after-failure probe metrics.
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import tyro


def _slug(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


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


def _summarize_run(run_dir: Path, label: str, persistence_enabled: bool) -> dict[str, object]:
    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "label": label,
        "persistence_enabled": persistence_enabled,
    }
    payload = _load_json(run_dir / "metrics.json")
    if payload is None:
        return summary

    metrics = payload.get("metrics", payload)
    psychometric = metrics.get("psychometric", {})
    chronometric = metrics.get("chronometric", {})
    history = metrics.get("history", {})
    probe = metrics.get("adaptive_control_probe", {})
    exploration_probe = metrics.get("exploration_probe", {})
    quality = metrics.get("quality", {})

    summary.update(
        {
            "psychometric_slope": psychometric.get("slope"),
            "psychometric_bias": psychometric.get("bias"),
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
    )
    weak_retry = summary.get("retry_after_failure_weak")
    strong_retry = summary.get("retry_after_failure_strong")
    if isinstance(weak_retry, (int, float)) and isinstance(strong_retry, (int, float)):
        summary["retry_gap"] = float(weak_retry) - float(strong_retry)
    weak_switch = summary.get("switch_after_failure_weak")
    strong_switch = summary.get("switch_after_failure_strong")
    if isinstance(weak_switch, (int, float)) and isinstance(strong_switch, (int, float)):
        summary["switch_gap"] = float(strong_switch) - float(weak_switch)
    weak_explore = summary.get("switch_after_streak_weak")
    strong_explore = summary.get("switch_after_streak_strong")
    if isinstance(weak_explore, (int, float)) and isinstance(strong_explore, (int, float)):
        summary["exploration_gap"] = float(weak_explore) - float(strong_explore)
    return summary


@dataclass(slots=True)
class SweepArgs:
    """Arguments for the persistence calibration sweep."""

    run_root: Path = Path("runs/adaptive_control_persistence_sweep")
    persistence_bias_scales: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0)
    persistence_learning_rates: Sequence[float] = (0.2, 0.4, 0.6, 0.8)
    include_no_persistence_control: bool = True
    seed: int = 42
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
    drift_scale: float = 10.0
    history_bias_scale: float = 2.0
    history_drift_scale: float = 0.3
    lapse_rate: float = 0.05
    dry_run: bool = False

    def run(self) -> None:
        """Execute the persistence sweep and write summaries."""
        self.run_root.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, object]] = []

        if self.include_no_persistence_control:
            run_dir = self.run_root / f"seed{self.seed}_no_persistence"
            train_cmd = self._build_train_command(
                run_dir=run_dir,
                persistence_enabled=False,
                persistence_bias_scale=0.0,
                persistence_learning_rate=0.0,
            )
            print(f"\n{'=' * 80}")
            print("Running no-persistence control")
            print(f"Output: {run_dir}")
            print(f"{'=' * 80}")
            _run_command(train_cmd, dry_run=self.dry_run)
            _run_command(self._build_eval_command(run_dir), dry_run=self.dry_run)
            if not self.dry_run:
                summaries.append(_summarize_run(run_dir, "no_persistence", False))

        for bias_scale, learning_rate in itertools.product(
            self.persistence_bias_scales,
            self.persistence_learning_rates,
        ):
            run_name = (
                f"seed{self.seed}_pb{_slug(bias_scale)}_plr{_slug(learning_rate)}"
            )
            run_dir = self.run_root / run_name
            print(f"\n{'=' * 80}")
            print(
                "Running persistence sweep: "
                f"persistence_bias_scale={bias_scale}, "
                f"persistence_learning_rate={learning_rate}"
            )
            print(f"Output: {run_dir}")
            print(f"{'=' * 80}")
            train_cmd = self._build_train_command(
                run_dir=run_dir,
                persistence_enabled=True,
                persistence_bias_scale=bias_scale,
                persistence_learning_rate=learning_rate,
            )
            _run_command(train_cmd, dry_run=self.dry_run)
            _run_command(self._build_eval_command(run_dir), dry_run=self.dry_run)
            if not self.dry_run:
                summary = _summarize_run(run_dir, run_name, True)
                summary["persistence_bias_scale"] = bias_scale
                summary["persistence_learning_rate"] = learning_rate
                summaries.append(summary)

        if self.dry_run or not summaries:
            return

        summaries.sort(
            key=lambda row: (
                float(row.get("retry_gap") or float("-inf")),
                float(row.get("psychometric_slope") or float("-inf")),
            ),
            reverse=True,
        )

        csv_path = self.run_root / "sweep_summary.csv"
        fieldnames = sorted({key for row in summaries for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)

        json_path = self.run_root / "sweep_summary.json"
        json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

        print(f"\n{'=' * 80}")
        print("Adaptive Control Persistence Sweep Summary")
        print(f"{'=' * 80}")
        print(
            f"{'label':<32} | {'retry':>7} | {'switch':>7} | {'explore':>7} | "
            f"{'psych':>7} | {'chrono':>8}"
        )
        print("-" * 80)
        for row in summaries:
            retry_gap = row.get("retry_gap")
            switch_gap = row.get("switch_gap")
            exploration_gap = row.get("exploration_gap")
            psych = row.get("psychometric_slope")
            chrono = row.get("chronometric_slope")
            retry_gap_str = f"{retry_gap:.3f}" if isinstance(retry_gap, (int, float)) else "n/a"
            switch_gap_str = f"{switch_gap:.3f}" if isinstance(switch_gap, (int, float)) else "n/a"
            exploration_gap_str = f"{exploration_gap:.3f}" if isinstance(exploration_gap, (int, float)) else "n/a"
            psych_str = f"{psych:.2f}" if isinstance(psych, (int, float)) else "n/a"
            chrono_str = f"{chrono:.2f}" if isinstance(chrono, (int, float)) else "n/a"
            print(
                f"{str(row.get('label')):<32} | {retry_gap_str:>7} | "
                f"{switch_gap_str:>7} | {exploration_gap_str:>7} | "
                f"{psych_str:>7} | {chrono_str:>8}"
            )

        print(f"\nSummary saved to {csv_path}")
        print(f"Summary JSON saved to {json_path}")

    def _build_train_command(
        self,
        *,
        run_dir: Path,
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
            str(self.seed),
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
        ]
        if not persistence_enabled:
            cmd.append("--no-persistence-enabled")
        return cmd
    @staticmethod
    def _build_eval_command(run_dir: Path) -> list[str]:
        return [
            sys.executable,
            "scripts/evaluate_agent.py",
            "--run",
            str(run_dir),
        ]


def main() -> None:
    """CLI entry point."""
    tyro.cli(SweepArgs).run()


if __name__ == "__main__":
    main()
