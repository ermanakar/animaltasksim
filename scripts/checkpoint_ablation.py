#!/usr/bin/env python3
"""Evaluate runtime control interventions on one immutable trained checkpoint."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import torch
import tyro

from agents.adaptive_control_config import AdaptiveControlConfig, AdaptiveControlProfile
from agents.adaptive_control_trainer import AdaptiveControlTrainer
from agents.losses import LossWeights
from scripts.adaptive_control_validation_suite import RUN_ARTIFACTS, ValidationSuiteArgs, _file_hash


@dataclass(slots=True)
class CheckpointAblationArgs:
    """Settings for acute interventions; no optimization is performed."""

    source_run: Path
    output_root: Path
    profiles: Sequence[AdaptiveControlProfile] = ("full_control", "persistence_only", "exploration_only", "no_control")
    paired_trial_seed: int = 42
    reference_log: Path | None = None
    episodes: int | None = None
    trials_per_episode: int | None = None

    def run(self) -> None:
        """Load identical weights for every profile into new output directories."""
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise RuntimeError("Checkpoint ablations require a new, empty output root.")
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("Profiles must be nonempty and unique.")
        source_hash = _file_hash(self.source_run / "model.pt")
        config_hash = _file_hash(self.source_run / "config.json")
        weights = torch.load(self.source_run / "model.pt", map_location="cpu", weights_only=True)
        baseline_inputs: dict[str, object] | None = None
        for profile in self.profiles:
            run_dir = self.output_root / profile
            config = self._config(run_dir, profile)
            inputs = ValidationSuiteArgs(task=config.task)._run_provenance([
                "checkpoint_ablation", "--reference-log", str(config.reference_log),
            ])
            if baseline_inputs is None:
                baseline_inputs = inputs
            elif inputs != baseline_inputs:
                raise RuntimeError("Source or reference changed between interventions.")
            trainer = AdaptiveControlTrainer(config)
            trainer.model.load_state_dict(weights, strict=True)
            trainer.model.eval()
            paths = config.output_paths()
            stats = trainer.rollout(paths, paired_trial_seed=self.paired_trial_seed)
            trainer.save(paths, training_metrics={}, rollout_stats=stats)
            subprocess.run([sys.executable, "-m", "scripts.evaluate_agent", "--run", str(run_dir)], check=True)
            if any(not torch.equal(value, weights[key]) for key, value in trainer.model.state_dict().items()):
                raise RuntimeError("Checkpoint parameters changed during evaluation.")
            if source_hash != _file_hash(self.source_run / "model.pt") or config_hash != _file_hash(self.source_run / "config.json"):
                raise RuntimeError("Source checkpoint/config changed during evaluation.")
            if inputs != ValidationSuiteArgs(task=config.task)._run_provenance([
                "checkpoint_ablation", "--reference-log", str(config.reference_log),
            ]):
                raise RuntimeError("Source or reference changed during intervention.")
            manifest = {
                "experiment_kind": "fixed_checkpoint_runtime_ablation",
                "source_checkpoint_sha256": source_hash,
                "source_config_sha256": config_hash,
                "profile": profile,
                "paired_trial_seed": self.paired_trial_seed,
                "randomness": "Trial-indexed separate DDM and lapse streams; environment seed shared across profiles. Action-dependent reward outcomes may differ.",
                "inputs": inputs,
                "outputs": {name: _file_hash(run_dir / name) for name in RUN_ARTIFACTS},
                "interpretation": "Acute computational effect in this checkpoint, not retraining, neural necessity, or animal validation.",
            }
            (run_dir / "checkpoint_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _config(self, run_dir: Path, profile: AdaptiveControlProfile) -> AdaptiveControlConfig:
        """Preserve source settings except explicit evaluation overrides."""
        payload = json.loads((self.source_run / "config.json").read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(AdaptiveControlConfig)}
        values = {key: value for key, value in payload.items() if key in allowed}
        for key in ("reference_log", "output_dir"):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        if isinstance(values.get("loss_weights"), dict):
            values["loss_weights"] = LossWeights(**values["loss_weights"])
        values.update(output_dir=run_dir, epochs=0)
        for key in ("reference_log", "episodes", "trials_per_episode"):
            if getattr(self, key) is not None:
                values[key] = getattr(self, key)
        config = AdaptiveControlConfig(**values)
        config.apply_control_profile(profile)
        return config


__all__ = ["CheckpointAblationArgs"]

if __name__ == "__main__":
    tyro.cli(CheckpointAblationArgs).run()
