"""Revision 1 of the within-animal recovery study: refit the switching models under a higher iteration limit."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from eval.within_animal import MODELS, Sessions, design, fit, predict
from eval.within_animal_recovery import CELLS, _subset, simulate
from scripts.recover_within_animal import load_schedules, summarize

__all__ = ["Args", "CARRIED", "REFITTED", "merge", "run", "score"]

CODE = (Path("scripts/recheck_within_animal_recovery.py"), Path("scripts/recover_within_animal.py"),
        Path("eval/within_animal.py"), Path("eval/within_animal_recovery.py"))
CARRIED = ("fixed", "history", "drift", "drift_history")
REFITTED = ("switching", "switching_history")
CHECK = "fixed"  # cell whose first seed is refitted with all six models


@dataclass(slots=True)
class Args:
    """`plan` freezes inputs and settings; `run` refits, resumably; `summarize` applies the unchanged gate once."""

    stage: Literal["plan", "run", "summarize"]
    first: Path = Path("runs/within_animal_recovery_v1")
    converted: Path = Path("runs/ibl_deep_converted_v1")
    output: Path = Path("runs/within_animal_recovery_v2")
    maxiter: int = 5000
    workers: int = 8


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(data: Sessions, models: tuple[str, ...], maxiter: int) -> dict[str, dict]:
    """Even/odd cross-fitted held-out loss per committed trial, as in the first study, with an explicit limit."""
    full = design(data)
    halves = [np.arange(0, len(data.length), 2), np.arange(1, len(data.length), 2)]
    committed = int(full.committed.sum())
    out: dict[str, dict] = {"nll": {}, "converged": {}}
    for model in models:
        total, converged = 0.0, []
        for train, test in (halves, halves[::-1]):
            fitted = fit(_subset(full, train, data.length), model, maxiter=maxiter)
            total += float(np.nansum(predict(_subset(full, test, data.length), fitted)))
            converged.append(fitted.converged)
        out["nll"][model] = total / committed
        out["converged"][model] = bool(all(converged))
    return out


def merge(first: list[dict], revised: list[dict]) -> list[dict]:
    """First-study animals with the refitted models' scores replaced; any mismatch in the synthetic data is an error."""
    merged = []
    for old, new in zip(first, revised, strict=True):
        if (old["subject"], old["trials"], old["truth"]) != (new["subject"], new["trials"], new["truth"]):
            raise RuntimeError(f"Regenerated data differ from the first study for {old['subject']}")
        if not all(old["converged"][model] for model in CARRIED):
            raise RuntimeError(f"A carried-over fit was not converged for {old['subject']}")
        merged.append(old | {"nll": old["nll"] | {m: new["nll"][m] for m in REFITTED},
                             "converged": old["converged"] | {m: new["converged"][m] for m in REFITTED}})
    return merged


def _plan(args: Args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    first = json.loads((args.first / "plan.json").read_text())
    plan = {"status": "revision 1: switching models refitted under a higher iteration limit; synthetic choices only",
            "protocol": "docs/WITHIN_ANIMAL_RECOVERY_REVISION_PROTOCOL.md", "maxiter": args.maxiter,
            "refitted": list(REFITTED), "carried": list(CARRIED), "check": f"{CHECK}_seed{first['seeds'][0]}",
            "first_plan_sha256": _digest(args.first / "plan.json"), "first_result_sha256": _digest(args.first / "result.json"),
            "first_cells_sha256": {p.name: _digest(p) for p in sorted((args.first / "cells").glob("*.json"))},
            "code_sha256": {str(p): _digest(p) for p in CODE}}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "plan.json"
    path.write_text(json.dumps(plan, indent=1) + "\n")
    path.with_suffix(".sha256").write_text(_digest(path) + "\n")
    print(json.dumps({"datasets": len(first["seeds"]) * len(first["cells"]), "maxiter": args.maxiter}))


def _frozen(args: Args) -> tuple[dict, dict]:
    path = args.output / "plan.json"
    if _digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Plan changed after freezing")
    plan = json.loads(path.read_text())
    if any(_digest(Path(p)) != value for p, value in plan["code_sha256"].items()):
        raise RuntimeError("Code changed after the plan was frozen")
    if any(_digest(args.first / "cells" / name) != value for name, value in plan["first_cells_sha256"].items()):
        raise RuntimeError("First-study outputs changed")
    if _digest(args.first / "plan.json") != plan["first_plan_sha256"]:
        raise RuntimeError("First-study plan changed")
    return plan, json.loads((args.first / "plan.json").read_text())


def _work(task: tuple[str, Sessions, str, int, int, tuple[str, ...], int]) -> dict[str, object]:
    name, schedule, cell, seed, index, models, maxiter = task
    simulated, truth = simulate(schedule, cell, np.random.default_rng([seed, list(CELLS).index(cell), index]))
    live = np.arange(simulated.choice.shape[1])[None, :] < simulated.length[:, None]
    trials = int((live & (simulated.choice != 0)).sum())
    return {"subject": name, "truth": truth, "trials": trials} | score(simulated, models, maxiter)


def _run(args: Args) -> None:
    plan, first = _frozen(args)
    schedules = load_schedules(args.converted, first["eligibility"]["min_trials"], first["eligibility"]["min_sessions"])
    if list(schedules) != list(first["animals"]):
        raise RuntimeError("Schedules differ from the first study's frozen plan")
    cells = args.output / "cells"
    cells.mkdir(exist_ok=True)
    jobs = [(cell, seed, REFITTED, f"{cell}_seed{seed}.json") for seed in first["seeds"] for cell in first["cells"]]
    jobs.insert(0, (CHECK, first["seeds"][0], MODELS, f"{plan['check']}_all_models.json"))
    with ProcessPoolExecutor(args.workers) as pool:
        for cell, seed, models, name in jobs:
            path = cells / name
            if path.exists():
                continue
            tasks = [(subject, schedule, cell, seed, index, models, plan["maxiter"])
                     for index, (subject, schedule) in enumerate(schedules.items())]
            animals = list(pool.map(_work, tasks, chunksize=1))
            partial = path.with_suffix(".partial")
            partial.write_text(json.dumps({"cell": cell, "seed": seed, "maxiter": plan["maxiter"], "animals": animals}, indent=1) + "\n")
            partial.replace(path)
            print(f"finished {name}", flush=True)


def _summarize(args: Args) -> None:
    path = args.output / "result.json"
    if path.exists():
        raise RuntimeError("Result exists; it is never replaced")
    plan, first = _frozen(args)
    old = lambda name: json.loads((args.first / "cells" / name).read_text())["animals"]  # noqa: E731
    new = lambda name: json.loads((args.output / "cells" / name).read_text())["animals"]  # noqa: E731
    seed0 = first["seeds"][0]
    check = max(abs(a["nll"][m] - b["nll"][m]) for a, b in zip(old(f"{plan['check']}.json"), new(f"{plan['check']}_all_models.json"),
                                                                 strict=True) for m in CARRIED)
    if check > 1e-12:
        raise RuntimeError(f"Carried-over models did not reproduce under the new limit (largest difference {check:.3e})")
    cells, changes = {}, {}
    for cell in first["cells"]:
        before = [old(f"{cell}_seed{seed}.json") for seed in first["seeds"]]
        cells[cell] = [merge(b, new(f"{cell}_seed{seed}.json")) for b, seed in zip(before, first["seeds"], strict=True)]
        pairs = [(a, b) for run_old, run_new in zip(before, cells[cell], strict=True) for a, b in zip(run_old, run_new, strict=True)]
        changes[cell] = {m: {"animals_changed": sum(abs(a["nll"][m] - b["nll"][m]) > 1e-9 for a, b in pairs),
                             "largest_change": max(abs(a["nll"][m] - b["nll"][m]) for a, b in pairs),
                             "nonconverged_before": sum(not a["converged"][m] for a, _ in pairs),
                             "nonconverged_after": sum(not b["converged"][m] for _, b in pairs)} for m in REFITTED}
    result = summarize(cells, old(f"drift_seed{seed0}_states81.json"), cells["drift"][0])
    result |= {"revision": 1, "maxiter": plan["maxiter"], "carryover_check": check, "changes": changes,
               "plan_sha256": _digest(args.output / "plan.json")}
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "criteria": result["criteria"], "carryover_check": check, "changes": changes,
                      "nonconverged": {c: r["nonconverged_fraction"] for c, r in result["cells"].items()},
                      "probabilities": {c: r["probabilities"] for c, r in result["cells"].items()}}, indent=1))


def run(args: Args) -> None:
    """Dispatch one explicit stage."""
    {"plan": _plan, "run": _run, "summarize": _summarize}[args.stage](args)


if __name__ == "__main__":
    run(tyro.cli(Args))
