"""What the fitted latent side bias predicts about response time, omissions and block changes (milestone 4)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from eval.within_animal import Sessions
from eval.within_animal_state import DIRECTIONS, MODEL, SIZES, animal_statistics, estimate_state, simulate_with_state, verdict
from scripts.compare_within_animal import load_sessions

__all__ = ["Args", "CHECK", "check_gate", "load_response_times", "read", "run"]

CODE = (Path("scripts/predict_from_state.py"), Path("scripts/compare_within_animal.py"), Path("eval/within_animal_state.py"),
        Path("eval/within_animal.py"), Path("eval/within_animal_recovery.py"))
PROTOCOL = Path("docs/STATE_PREDICTIONS_PROTOCOL.md")
CHECK = {"null": False, "effect": True}  # synthetic set -> whether response times and omissions follow the state


@dataclass(slots=True)
class Args:
    """`plan` freezes; `check` runs the synthetic method check; `run` analyses real data silently; `summarize` reads once."""

    stage: Literal["plan", "check", "run", "summarize"]
    converted: Path = Path("runs/ibl_deep_converted_v1")
    output: Path = Path("runs/state_predictions_v1")
    min_trials: int = 150
    min_sessions: int = 6
    maxiter: int = 5000
    resamples: int = 10000
    workers: int = 8


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_response_times(converted: Path, min_trials: int, min_sessions: int) -> dict[str, np.ndarray]:
    """Response time in ms per animal as [session, trial], NaN where missing, in the session order of `load_sessions`."""
    manifest = json.loads((converted / "manifest.json").read_text())
    eligible = {s["id"]: s["session_number"] for s in manifest["sessions"]
                if s["status"] == "converted" and s["family"] == "biased" and s["trials"] >= min_trials}
    out = {}
    for name in sorted(manifest["files"]):
        if not name.startswith("biased/"):
            continue
        rows: dict[str, list[float]] = {}
        with (converted / name).open() as lines:
            for line in lines:
                trial = json.loads(line)
                if trial["session_id"] in eligible:
                    value = trial.get("rt_ms")
                    rows.setdefault(trial["session_id"], []).append(np.nan if value is None else float(value))
        if len(rows) < min_sessions:
            continue
        ordered = [rows[eid] for eid in sorted(rows, key=eligible.__getitem__)]
        rt = np.full((len(ordered), max(len(r) for r in ordered)), np.nan)
        for row, values in enumerate(ordered):
            rt[row, :len(values)] = values
        out[Path(name).stem] = rt
    return out


def _plan(args: Args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    animals = load_sessions(args.converted, args.min_trials, args.min_sessions)
    plan = {"status": "exploratory: development animals; associations between a fitted latent bias and measures not used in fitting",
            "protocol": str(PROTOCOL), "protocol_sha256": _digest(PROTOCOL), "model": MODEL, "sizes_of_interest": SIZES,
            "directions": DIRECTIONS, "maxiter": args.maxiter, "resamples": args.resamples, "check_sets": CHECK,
            "eligibility": {"family": "biased", "min_trials": args.min_trials, "min_sessions": args.min_sessions},
            "animals": {name: {"lab": lab, "sessions": len(s.length), "trials": int(s.length.sum())}
                        for name, (lab, s) in animals.items()},
            "converted_manifest_sha256": _digest(args.converted / "manifest.json"),
            "code_sha256": {str(p): _digest(p) for p in CODE}}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "plan.json"
    path.write_text(json.dumps(plan, indent=1) + "\n")
    path.with_suffix(".sha256").write_text(_digest(path) + "\n")
    print(json.dumps({"animals": len(animals)}))


def _frozen(args: Args) -> dict:
    path = args.output / "plan.json"
    if _digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Plan changed after freezing")
    plan = json.loads(path.read_text())
    if any(_digest(Path(p)) != value for p, value in plan["code_sha256"].items()):
        raise RuntimeError("Code changed after the plan was frozen")
    if _digest(PROTOCOL) != plan["protocol_sha256"]:
        raise RuntimeError("Protocol changed after the plan was frozen")
    return plan


def _work(task: tuple[str, Sessions, np.ndarray | None, int, bool | None, int]) -> dict[str, object]:
    """One animal: real data when `effect` is None, otherwise synthetic choices and response times on its schedule."""
    name, data, rt, maxiter, effect, index = task
    if effect is not None:
        data, rt = simulate_with_state(data, np.random.default_rng([20261125, int(effect), index]), effect)
    state = estimate_state(data, maxiter=maxiter)
    return {"subject": name, "converged": bool(state["converged"])} | animal_statistics(data, rt, state)


def _map(args: Args, tasks: list) -> list[dict]:
    if args.workers <= 1:
        return [_work(task) for task in tasks]
    with ProcessPoolExecutor(args.workers) as pool:
        return list(pool.map(_work, tasks, chunksize=1))


def read(animals: list[dict], rng: np.random.Generator, resamples: int) -> dict[str, dict]:
    """The protocol's reading of each quantity, over the animals that have it."""
    return {key: verdict(np.array([a[key] for a in animals if a[key] is not None]), key, rng, resamples) for key in SIZES}


def check_gate(readings: dict[str, dict[str, dict]]) -> bool:
    """Nothing is read into the null set, and the three planted effects are found."""
    quiet = all(r["reading"].startswith("not supported") for r in readings["null"].values())
    found = all(readings["effect"][key]["reading"] == "supported" for key in ("p1_congruence", "p2_magnitude", "p3_omission"))
    return bool(quiet and found)


def _check(args: Args) -> None:
    path = args.output / "check.json"
    if path.exists():
        raise RuntimeError("Check exists; it is never replaced")
    plan = _frozen(args)
    schedules = load_sessions(args.converted, args.min_trials, args.min_sessions)
    rng, readings = np.random.default_rng(20261125), {}
    for name, effect in plan["check_sets"].items():
        tasks = [(subject, data, None, plan["maxiter"], effect, index) for index, (subject, (_, data)) in enumerate(schedules.items())]
        readings[name] = read(_map(args, tasks), rng, plan["resamples"])
    result = {"passed": check_gate(readings), "sets": readings, "plan_sha256": _digest(args.output / "plan.json")}
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps(result, indent=1))


def _run(args: Args) -> None:
    plan = _frozen(args)
    check = args.output / "check.json"
    if not check.exists() or not json.loads(check.read_text())["passed"]:
        raise RuntimeError("The synthetic method check has not passed; the real analysis does not run")
    path = args.output / "animals.json"
    if path.exists():
        raise RuntimeError("Per-animal output exists; it is never replaced")
    animals = load_sessions(args.converted, args.min_trials, args.min_sessions)
    times = load_response_times(args.converted, args.min_trials, args.min_sessions)
    if list(animals) != list(plan["animals"]) or list(times) != list(animals):
        raise RuntimeError("Sessions differ from the frozen plan")
    tasks = []
    for index, (subject, (_, data)) in enumerate(animals.items()):
        if times[subject].shape != data.contrast.shape:
            raise RuntimeError(f"Response times do not align with trials for {subject}")
        tasks.append((subject, data, times[subject], plan["maxiter"], None, index))
    path.write_text(json.dumps({"animals": _map(args, tasks)}, indent=1) + "\n")
    print("finished", flush=True)  # deliberately no statistic


def _summarize(args: Args) -> None:
    path = args.output / "result.json"
    if path.exists():
        raise RuntimeError("Result exists; it is never replaced")
    plan = _frozen(args)
    animals = json.loads((args.output / "animals.json").read_text())["animals"]
    labs = {name: record["lab"] for name, record in plan["animals"].items()}
    largest = max(sorted(set(labs.values())), key=list(labs.values()).count)
    rng = np.random.default_rng(20261125)
    result = {"status": plan["status"], "nonconverged_animals": sum(not a["converged"] for a in animals),
              "readings": read(animals, rng, plan["resamples"]),
              "without_largest_lab": {"lab_removed": largest} | read(
                  [a for a in animals if labs[a["subject"]] != largest], rng, plan["resamples"]),
              "block_late": {"mean": float(np.mean([a["block_late"] for a in animals if a["block_late"] is not None]))},
              "totals": {key: int(sum(a[key] for a in animals)) for key in ("rt_trials", "omissions", "block_changes")},
              "check_passed": json.loads((args.output / "check.json").read_text())["passed"],
              "plan_sha256": _digest(args.output / "plan.json")}
    with (args.output / "subject_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        columns = [*SIZES, "block_late", "rt_trials", "omissions", "block_changes"]
        writer.writerow(["subject", "lab", *columns])
        for a in animals:
            writer.writerow([a["subject"], labs[a["subject"]], *("" if a[c] is None else a[c] for c in columns)])
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps(result, indent=1))


def run(args: Args) -> None:
    """Dispatch one explicit stage."""
    {"plan": _plan, "check": _check, "run": _run, "summarize": _summarize}[args.stage](args)


if __name__ == "__main__":
    run(tyro.cli(Args))
