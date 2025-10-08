"""Drift-diffusion (DDM) baseline for the RDM task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from animaltasksim.config import ProjectPaths
from animaltasksim.seeding import seed_everything
from envs.rdm_macaque import ACTION_HOLD, ACTION_LEFT, ACTION_RIGHT, AgentMetadata, RDMConfig, RDMMacaqueEnv


@dataclass(slots=True)
class DDMConfig:
    trials_per_episode: int = 400
    episodes: int = 1
    seed: int = 1234
    drift_gain: float = 0.1
    noise: float = 1.0
    bound: float = 1.0
    non_decision_ms: float = 100.0
    per_step_cost: float = 0.0
    output_dir: Path = field(default_factory=lambda: ProjectPaths.from_cwd().runs / "rdm_ddm")

    def output_paths(self) -> dict[str, Path]:
        out = Path(self.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return {
            "root": out,
            "config": out / "config.json",
            "log": out / "trials.ndjson",
            "metrics": out / "metrics.json",
        }


class DDMBaseline:
    def __init__(self, config: DDMConfig) -> None:
        self.config = config

    def sample_dt_ms(self) -> float:
        return 10.0

    def run(self) -> dict[str, object]:
        paths = self.config.output_paths()
        env_config = RDMConfig(
            trials_per_episode=self.config.trials_per_episode,
            per_step_cost=self.config.per_step_cost,
            log_path=paths["log"],
            agent=AgentMetadata(name="ddm_baseline", version="0.1.0"),
            seed=self.config.seed,
            evidence_gain=self.config.drift_gain,
            momentary_sigma=self.config.noise,
            collapsing_bound=False,
        )
        env = RDMMacaqueEnv(env_config)

        seed_everything(self.config.seed)

        decisions = []
        rts = []

        for episode in range(self.config.episodes):
            observation, info = env.reset(seed=self.config.seed + episode)
            terminated = False
            evidence = 0.0
            steps = 0

            while not terminated:
                phase = info["phase"]
                if phase == "response":
                    evidence += float(observation.get("coherence", 0.0))
                    if evidence >= self.config.bound:
                        action = ACTION_RIGHT
                    elif evidence <= -self.config.bound:
                        action = ACTION_LEFT
                    else:
                        action = ACTION_HOLD
                    steps += 1
                else:
                    action = ACTION_HOLD

                observation, reward, terminated, truncated, info = env.step(action)
                if info["phase"] == "outcome" and info["phase_step"] == 0:
                    decisions.append(reward)
                    rts.append(max(steps, 1) * self.sample_dt_ms() + self.config.non_decision_ms)
                    evidence = 0.0
                    steps = 0

        env.close()

        metrics = {
            "episodes": self.config.episodes,
            "mean_reward": float(np.mean(decisions) if decisions else 0.0),
            "mean_rt_ms": float(np.mean(rts) if rts else 0.0),
        }
        paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._persist_config(paths["config"], env_config)
        return metrics

    def _persist_config(self, path: Path, env_config: RDMConfig) -> None:
        config_dict = asdict(self.config)
        config_dict["output_dir"] = str(self.config.output_dir)
        payload = {
            "ddm": config_dict,
            "environment": {
                "trials_per_episode": env_config.trials_per_episode,
                "per_step_cost": env_config.per_step_cost,
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["DDMBaseline", "DDMConfig"]
