"""Recovery study for within-animal models on the real schedules of the deep reference."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from eval.within_animal import MODELS, Sessions, pad
from eval.within_animal_recovery import (CELLS, CONTRASTS_TESTED, DELTA, cohort_probabilities, decide, score_animal,
                                         simulate)

__all__ = ["Args", "load_schedules", "summarize", "run"]

CODE = (Path("scripts/recover_within_animal.py"), Path("eval/within_animal.py"), Path("eval/within_animal_recovery.py"))
STATE_CELLS = {"drift": "drift", "drift_history": "drift", "switching": "switching", "switching_history": "switching"}


@dataclass(slots=True)
class Args:
    """`plan` freezes inputs and settings; `run` simulates and fits, resumably; `summarize` applies the gate once."""

    stage: Literal["plan", "run", "summarize"]
    converted: Path = Path("runs/ibl_deep_converted_v1")
    output: Path = Path("runs/within_animal_recovery_v1")
    seeds: tuple[int, ...] = (0, 1, 2)
    min_trials: int = 150
    min_sessions: int = 6
    workers: int = 8


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_schedules(converted: Path, min_trials: int, min_sessions: int) -> dict[str, Sessions]:
    """Contrasts, priors and omission positions per animal, in session order. Choices and outcomes are not read."""
    manifest = json.loads((converted / "manifest.json").read_text())
    order = {s["id"]: s["session_number"] for s in manifest["sessions"]
             if s["status"] == "converted" and s["family"] == "biased" and s["trials"] >= min_trials}
    schedules = {}
    for name, record in sorted(manifest["files"].items()):
        if not name.startswith("biased/"):
            continue
        if _digest(converted / name) != record["sha256"]:
            raise RuntimeError(f"Converted file changed: {name}")
        rows: dict[str, list[tuple[float, float, float]]] = {}
        with (converted / name).open() as lines:
            for line in lines:
                trial = json.loads(line)
                if trial["session_id"] in order:
                    # A committed trial is marked 1 as a placeholder; the animal's actual side is discarded.
                    rows.setdefault(trial["session_id"], []).append(
                        (trial["stimulus"]["contrast"], trial["block_prior"]["p_right"], float(trial["action"] != 2)))
        if len(rows) < min_sessions:
            continue
        sessions = []
        for eid in sorted(rows, key=order.__getitem__):
            values = np.asarray(rows[eid])
            sessions.append({"contrast": values[:, 0], "prior": values[:, 1], "choice": values[:, 2],
                             "correct": np.zeros(len(values))})
        schedules[Path(name).stem] = pad(sessions)
    return schedules


def _plan(args: Args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    schedules = load_schedules(args.converted, args.min_trials, args.min_sessions)
    plan = {"status": "synthetic choices on real schedules; no real choice or outcome is read",
            "protocol": "docs/WITHIN_ANIMAL_RECOVERY_PROTOCOL.md", "cells": {k: list(v) for k, v in CELLS.items()},
            "seeds": list(args.seeds), "delta": DELTA, "models": list(MODELS), "contrasts": CONTRASTS_TESTED,
            "eligibility": {"family": "biased", "min_trials": args.min_trials, "min_sessions": args.min_sessions},
            "animals": {name: {"sessions": len(s.length), "trials": int(s.length.sum())} for name, s in schedules.items()},
            "converted_manifest_sha256": _digest(args.converted / "manifest.json"),
            "code_sha256": {str(p): _digest(p) for p in CODE}}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "plan.json"
    path.write_text(json.dumps(plan, indent=1) + "\n")
    path.with_suffix(".sha256").write_text(_digest(path) + "\n")
    counts = [a["sessions"] for a in plan["animals"].values()], [a["trials"] for a in plan["animals"].values()]
    print(json.dumps({"animals": len(schedules), "median_sessions": float(np.median(counts[0])),
                      "median_trials": float(np.median(counts[1])), "total_trials": int(np.sum(counts[1]))}))


def _frozen(args: Args) -> dict:
    path = args.output / "plan.json"
    if _digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Plan changed after freezing")
    plan = json.loads(path.read_text())
    if any(_digest(Path(p)) != value for p, value in plan["code_sha256"].items()):
        raise RuntimeError("Code changed after the plan was frozen")
    return plan


def _work(task: tuple[str, Sessions, str, int, int, int]) -> dict[str, object]:
    name, schedule, cell, seed, index, states = task
    rng = np.random.default_rng([seed, list(CELLS).index(cell), index])
    simulated, truth = simulate(schedule, cell, rng)
    models = MODELS if states == 41 else ("drift", "drift_history")
    return {"subject": name, "truth": truth} | score_animal(simulated, states=states, models=models)


def _run(args: Args) -> None:
    plan = _frozen(args)
    schedules = load_schedules(args.converted, args.min_trials, args.min_sessions)
    if list(schedules) != list(plan["animals"]):
        raise RuntimeError("Schedules differ from the frozen plan")
    cells = args.output / "cells"
    cells.mkdir(exist_ok=True)
    jobs = [(cell, seed, 41) for seed in plan["seeds"] for cell in plan["cells"]] + [("drift", plan["seeds"][0], 81)]
    with ProcessPoolExecutor(args.workers) as pool:
        for cell, seed, states in jobs:
            path = cells / (f"{cell}_seed{seed}.json" if states == 41 else f"{cell}_seed{seed}_states{states}.json")
            if path.exists():
                continue
            tasks = [(name, schedule, cell, seed, index, states) for index, (name, schedule) in enumerate(schedules.items())]
            animals = list(pool.map(_work, tasks, chunksize=1))
            partial = path.with_suffix(".partial")
            partial.write_text(json.dumps({"cell": cell, "seed": seed, "states": states, "animals": animals}, indent=1) + "\n")
            partial.replace(path)
            print(f"finished {path.name}", flush=True)


def _gains(animals: list[dict]) -> np.ndarray:
    return np.array([[a["nll"][before] - a["nll"][after] for before, after in CONTRASTS_TESTED.values()] for a in animals])


def summarize(cells: dict[str, list[list[dict]]], fine: list[dict], coarse: list[dict], seed: int = 20261124) -> dict:
    """Detection probabilities per cell and the seven gate criteria of the protocol."""
    rng = np.random.default_rng(seed)
    report: dict[str, dict] = {}
    for cell, runs in cells.items():
        pooled = [a for run in runs for a in run]
        gains = _gains(pooled)
        best = [min(a["nll"], key=a["nll"].get) for a in pooled]
        low_high = [np.quantile(gains[rng.integers(len(gains), size=(2000, len(gains)))].mean(1), [0.025, 0.975], axis=0)]
        report[cell] = {
            "animals": len(pooled), "probabilities": cohort_probabilities(gains, rng),
            "per_seed_decisions": [decide(_gains(run), rng) for run in runs],
            "mean_gain": {name: {"mean": float(gains[:, i].mean()), "interval": [float(low_high[0][0, i]), float(low_high[0][1, i])],
                                 "animals_positive": int((gains[:, i] > 0).sum())}
                          for i, name in enumerate(CONTRASTS_TESTED)},
            "best_model_counts": {model: best.count(model) for model in MODELS},
            "nonconverged_fraction": float(np.mean([not ok for a in pooled for ok in a["converged"].values()])),
            "fitted_drift": {model: {key: float(np.median([a["drift"][model][key] for a in pooled])) for key in ("rho", "sd")}
                             for model in ("drift", "drift_history")},
            "true_drift_sd_median": float(np.median([a["truth"]["drift_sd"] for a in pooled]))}
    p = {cell: report[cell]["probabilities"] for cell in report}
    resolution = float(np.mean([max(abs(c["nll"][m] - f["nll"][m]) for m in ("drift", "drift_history"))
                                for c, f in zip(coarse, fine, strict=True)]))
    form_ok = all(p[c][f"form_{true}"] >= 0.90 and p[c]["form_drift" if true == "switching" else "form_switching"] <= 0.05
                  for c, true in STATE_CELLS.items())
    criteria = {
        "1_state_false_detection": max(p[c]["state"] for c in ("fixed", "history")) <= 0.05,
        "2_state_power": min(p[c]["state"] for c in STATE_CELLS) >= 0.90,
        "3_history_false_detection": max(p[c]["history"] for c in ("fixed", "drift", "switching")) <= 0.05,
        "4_history_power": min(p[c]["history"] for c in ("history", "drift_history", "switching_history")) >= 0.90,
        "5_form": form_ok,
        "6_convergence": max(report[c]["nonconverged_fraction"] for c in report) <= 0.02,
        "7_numerical_resolution": resolution <= 2e-4}
    return {"passed": all(criteria.values()), "criteria": criteria, "numerical_resolution": resolution, "cells": report}


def _summarize(args: Args) -> None:
    path = args.output / "result.json"
    if path.exists():
        raise RuntimeError("Result exists; it is never replaced")
    plan = _frozen(args)
    load = lambda name: json.loads((args.output / "cells" / name).read_text())["animals"]  # noqa: E731
    cells = {cell: [load(f"{cell}_seed{seed}.json") for seed in plan["seeds"]] for cell in plan["cells"]}
    first = plan["seeds"][0]
    result = summarize(cells, load(f"drift_seed{first}_states81.json"), cells["drift"][0])
    result["plan_sha256"] = _digest(args.output / "plan.json")
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "criteria": result["criteria"],
                      "numerical_resolution": result["numerical_resolution"],
                      "probabilities": {c: r["probabilities"] for c, r in result["cells"].items()}}, indent=1))


def run(args: Args) -> None:
    """Dispatch one explicit stage."""
    {"plan": _plan, "run": _run, "summarize": _summarize}[args.stage](args)


if __name__ == "__main__":
    run(tyro.cli(Args))
