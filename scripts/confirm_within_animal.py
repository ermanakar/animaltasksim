"""Confirmation of the within-animal claims on animals whose trials the project has never downloaded (milestone 5)."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import contextlib
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from eval.within_animal import MODELS
from eval.within_animal_recovery import CONTRASTS_TESTED, DELTA
from eval.within_animal_state import DIRECTIONS, SIZES
from scripts import compare_within_animal as comparison
from scripts import convert_ibl_deep_reference as conversion
from scripts import fetch_ibl_deep_reference as fetch
from scripts import predict_from_state as prediction

__all__ = ["Args", "CLAIMS", "LEVEL", "MIN_ANIMALS", "bounds", "read_claims", "run", "select_cohort", "summarize"]

CODE = tuple(Path(p) for p in (
    "scripts/confirm_within_animal.py", "scripts/compare_within_animal.py", "scripts/predict_from_state.py",
    "scripts/fetch_ibl_deep_reference.py", "scripts/convert_ibl_deep_reference.py", "eval/within_animal.py",
    "eval/within_animal_recovery.py", "eval/within_animal_state.py", "eval/ibl_deep_convert.py", "eval/schema_validator.py"))
PROTOCOL = Path("docs/CONFIRMATION_PROTOCOL.md")
STATUS = "confirmatory, internally frozen: animals whose trials were never downloaded by this project"
LEVEL, MIN_ANIMALS, MIN_SESSIONS, SEED = 0.99, 30, 6, 20261126
# claim -> (kind, quantities, how many of them must pass)
CLAIMS = {"c1_state_beyond_history": ("score", ("state_drift", "state_switching"), "any"),
          "c2_history_beyond_state": ("score", ("history_drift", "history_switching"), "all"),
          "p1_congruence": ("state", ("p1_congruence",), "all"),
          "p2_magnitude": ("state", ("p2_magnitude",), "all"),
          "p3_omission": ("state", ("p3_omission",), "all")}
SCORE_PAIRS = CONTRASTS_TESTED | comparison.DESCRIPTIVE


@dataclass(slots=True)
class Args:
    """Stages run separately. `cohort` and `plan` read saved metadata only; `download` is the first access to trials."""

    stage: Literal["cohort", "plan", "download", "convert", "check", "run", "summarize"]
    frame: Path = Path("runs/ibl_replication_plan/sampling_frame.json")
    reserved_manifest: Path = Path("runs/ibl_replication_plan/confirmation_manifest.json")
    development_manifest: Path = Path("runs/ibl_source_audit/source_manifest.json")
    output: Path = Path("runs/confirmation_v1")
    min_trials: int = 150
    maxiter: int = 5000
    resamples: int = 10000
    workers: int = 8
    shard_index: int = 0
    shard_count: int = 1


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cohort(frame: list[dict], excluded: set[str], min_sessions: int = MIN_SESSIONS) -> list[dict]:
    """Standard biased sessions of every animal outside `excluded` that has at least `min_sessions` of them."""
    sessions: dict[str, dict] = {}
    for item in frame:
        if fetch.protocol_family(item.get("task_protocol")) != "biased" or not item.get("lab") or item["subject"] in excluded:
            continue
        entry = {key: item[key] for key in fetch.IDENTITY_KEYS} | {"family": "biased"}
        if sessions.setdefault(str(item["id"]), entry) != entry:
            raise ValueError("Conflicting duplicate session identity.")
    count = Counter(s["subject"] for s in sessions.values())
    return sorted((s for s in sessions.values() if count[s["subject"]] >= min_sessions),
                  key=lambda s: (s["subject"], s["start_time"], s["id"]))


def _excluded(args: Args) -> set[str]:
    reserved = json.loads(args.reserved_manifest.read_text())
    development = {str(s["subject"]) for s in json.loads(args.development_manifest.read_text())["sessions"]}
    return {str(s) for s in reserved["excluded_subjects"]} | {str(s["subject"]) for s in reserved["sessions"]} | development


def _cohort(args: Args) -> None:
    path = args.output / "cohort.json"
    if path.exists():
        raise RuntimeError("Cohort exists; it is never replaced")
    sessions = select_cohort(json.loads(args.frame.read_text()), _excluded(args))
    labs = Counter({s["subject"]: s["lab"] for s in sessions}.values())
    cohort = {"status": "metadata only; no trial requested", "rule": f"standard biasedChoiceWorld; at least {MIN_SESSIONS} sessions; "
              "not a development or replication animal", "sessions": sessions,
              "counts": {"animals": sum(labs.values()), "sessions": len(sessions), "animals_by_lab": dict(sorted(labs.items()))},
              "input_sha256": {str(p): _digest(p) for p in (args.frame, args.reserved_manifest, args.development_manifest)}}
    args.output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cohort, indent=1, sort_keys=True) + "\n")
    print(json.dumps(cohort["counts"], indent=1))


def _plan(args: Args) -> None:
    path = args.output / "plan.json"
    if path.exists():
        raise RuntimeError("Plan exists; it is never replaced")
    cohort = json.loads((args.output / "cohort.json").read_text())
    if {s["subject"] for s in cohort["sessions"]} & _excluded(args):
        raise RuntimeError("A development or replication animal is in the cohort")
    plan = {"status": STATUS, "protocol": str(PROTOCOL), "protocol_sha256": _digest(PROTOCOL),
            "cohort_sha256": _digest(args.output / "cohort.json"), "cohort_counts": cohort["counts"],
            "claims": {name: list(rule) for name, rule in CLAIMS.items()}, "level": LEVEL, "delta": DELTA,
            "sizes_of_interest": SIZES, "directions": DIRECTIONS, "minimum_animals": MIN_ANIMALS,
            "eligibility": {"family": "biased", "min_trials": args.min_trials, "min_sessions": MIN_SESSIONS},
            "models": list(MODELS), "maxiter": args.maxiter, "resamples": args.resamples, "seed": SEED,
            "jobs": {name: [split, states, list(models)] for name, (split, states, models) in comparison.JOBS.items()},
            "check_sets": prediction.CHECK, "code_sha256": {str(p): _digest(p) for p in CODE}}
    source = args.output / "source"
    source.mkdir()
    acquisition = source / "plan.json"  # the layout the acquisition functions expect
    acquisition.write_text(json.dumps({"status": STATUS, "sessions": cohort["sessions"]}, indent=1, sort_keys=True) + "\n")
    acquisition.with_suffix(".sha256").write_text(_digest(acquisition) + "\n")
    path.write_text(json.dumps(plan, indent=1) + "\n")
    path.with_suffix(".sha256").write_text(_digest(path) + "\n")
    print(json.dumps(plan["cohort_counts"], indent=1))


def _frozen(args: Args) -> dict:
    path = args.output / "plan.json"
    if _digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Plan changed after freezing")
    plan = json.loads(path.read_text())
    if any(_digest(Path(p)) != value for p, value in plan["code_sha256"].items()):
        raise RuntimeError("Code changed after the plan was frozen")
    if _digest(PROTOCOL) != plan["protocol_sha256"]:
        raise RuntimeError("Protocol changed after the plan was frozen")
    if _digest(args.output / "cohort.json") != plan["cohort_sha256"]:
        raise RuntimeError("Cohort changed after the plan was frozen")
    return plan


def _source(args: Args, stage: str) -> fetch.Args:
    return fetch.Args(stage=stage, output=args.output / "source", shard_index=args.shard_index, shard_count=args.shard_count)  # type: ignore[arg-type]


def _download(args: Args) -> None:
    _frozen(args)
    fetch.run(_source(args, "download"))


def _convert(args: Args) -> None:
    """Verify the cache, convert with the frozen structural rules, and print counts that involve no behaviour."""
    _frozen(args)
    with contextlib.redirect_stdout(io.StringIO()):  # the reused stages print a performance-based count
        fetch.run(_source(args, "verify"))
        conversion.run(conversion.Args(stage="convert", source=args.output / "source", output=args.output / "converted"))
    source = json.loads((args.output / "source" / "source_manifest.json").read_text())["counts"]
    animals = comparison.load_sessions(args.output / "converted", args.min_trials, MIN_SESSIONS)
    print(json.dumps({"cached": source["cached"], "unavailable": source["unavailable"], "eligible_animals": len(animals),
                      "eligible_sessions": sum(len(s.length) for _, s in animals.values())}, indent=1))


def _animals(args: Args) -> dict:
    return comparison.load_sessions(args.output / "converted", args.min_trials, MIN_SESSIONS)


def _check(args: Args) -> None:
    """The synthetic method check of the state predictions, on the confirmation schedules; no real response is read."""
    path = args.output / "check.json"
    if path.exists():
        raise RuntimeError("Check exists; it is never replaced")
    plan = _frozen(args)
    rng, readings = np.random.default_rng(SEED), {}
    for name, effect in plan["check_sets"].items():
        tasks = [(subject, data, None, plan["maxiter"], effect, index) for index, (subject, (_, data)) in enumerate(_animals(args).items())]
        readings[name] = prediction.read(prediction._map(args, tasks), rng, plan["resamples"])  # type: ignore[arg-type]
    result = {"passed": prediction.check_gate(readings), "sets": readings, "plan_sha256": _digest(args.output / "plan.json")}
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps(result, indent=1))


def _run(args: Args) -> None:
    """Fit everything and print no score. The state measures are computed only if the method check passed."""
    plan = _frozen(args)
    check = args.output / "check.json"
    if not check.exists():
        raise RuntimeError("Run the synthetic method check first")
    animals = _animals(args)
    scores = args.output / "scores"
    scores.mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for name, (split, states, models) in plan["jobs"].items():
            path = scores / f"{name}.json"
            if path.exists():
                continue
            tasks = [(subject, data, split, states, tuple(models), plan["maxiter"]) for subject, (_, data) in animals.items()]
            partial = path.with_suffix(".partial")
            partial.write_text(json.dumps({"job": name, "animals": list(pool.map(comparison._work, tasks, chunksize=1))}, indent=1) + "\n")
            partial.replace(path)
            print(f"finished {name}", flush=True)
    path = args.output / "animals.json"
    if json.loads(check.read_text())["passed"] and not path.exists():
        times = prediction.load_response_times(args.output / "converted", args.min_trials, MIN_SESSIONS)
        if list(times) != list(animals) or any(times[s].shape != data.contrast.shape for s, (_, data) in animals.items()):
            raise RuntimeError("Response times do not align with trials")
        tasks = [(subject, data, times[subject], plan["maxiter"], None, index) for index, (subject, (_, data)) in enumerate(animals.items())]
        path.write_text(json.dumps({"animals": prediction._map(args, tasks)}, indent=1) + "\n")  # type: ignore[arg-type]
    print("finished", flush=True)  # deliberately no statistic


def bounds(values: np.ndarray, rng: np.random.Generator, resamples: int, level: float) -> dict[str, object]:
    """Equal-animal mean of each column with a percentile bootstrap interval over animals."""
    values = np.asarray(values, dtype=float).reshape(len(values), -1)
    means = values[rng.integers(len(values), size=(resamples, len(values)))].mean(1)
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return {"animals": len(values), "mean": values.mean(0).tolist(), "low": low.tolist(), "high": high.tolist()}


def read_claims(gains: dict[str, np.ndarray], values: dict[str, np.ndarray] | None, rng: np.random.Generator,
                resamples: int, level: float = LEVEL) -> dict[str, dict]:
    """The protocol's reading of each claim. `gains` are held-out score gains; `values` are the state quantities.

    Claims about the state are skipped when `values` is None, and a quantity no animal has is not confirmed.
    """
    out = {}
    for claim, (kind, names, need) in CLAIMS.items():
        if kind == "state" and values is None:
            continue
        parts, passed, against = {}, [], []
        for name in names:
            data = np.asarray((gains if kind == "score" else values)[name], dtype=float)
            if not len(data):
                parts[name], passed, against = {"animals": 0}, [*passed, False], [*against, False]
                continue
            sign, threshold = (1, DELTA) if kind == "score" else (DIRECTIONS[name], SIZES[name])
            b = bounds(sign * data, rng, resamples, level)  # signed so that the predicted direction is positive
            low, high = b["low"][0], b["high"][0]
            parts[name] = {"animals": b["animals"], "mean": sign * b["mean"][0], "interval": sorted([sign * low, sign * high]),
                           "animals_in_predicted_direction": int((sign * data > 0).sum())}
            passed.append(low > threshold)
            against.append(high < 0)
        confirmed = any(passed) if need == "any" else all(passed)
        out[claim] = {"reading": "confirmed" if confirmed else "contradicted" if all(against) else "not confirmed", "quantities": parts}
    return out


def _columns(animals: list[dict]) -> dict[str, np.ndarray]:
    table = comparison._table(animals, SCORE_PAIRS)
    return {name: table[:, i] for i, name in enumerate(SCORE_PAIRS)}


def _values(animals: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.array([a[key] for a in animals if a[key] is not None], dtype=float) for key in SIZES}


def summarize(plan: dict, labs: dict[str, str], jobs: dict[str, list[dict]], state: list[dict] | None) -> dict:
    """Run conditions, the five readings, the planned sensitivity analyses and the descriptive tables."""
    rng = np.random.default_rng(plan["seed"])
    resamples, primary = plan["resamples"], jobs["alternating"]
    descriptive = comparison.summarize({"status": plan["status"], "resamples": resamples,
                                        "animals": {name: {"lab": lab} for name, lab in labs.items()}}, jobs, seed=plan["seed"])
    conditions = {"eligible_animals": len(primary), "enough_animals": len(primary) >= plan["minimum_animals"],
                  "nonconverged_fraction": descriptive["conditions"]["nonconverged_fraction"],
                  "numerical_resolution": descriptive["conditions"]["numerical_resolution"],
                  "fits_met": descriptive["conditions"]["met"], "method_check_passed": state is not None}
    largest = max(sorted(set(labs.values())), key=list(labs.values()).count)
    keep = [labs[a["subject"]] != largest for a in primary]

    def reading(score_animals: list[dict], state_animals: list[dict] | None, level: float) -> dict[str, dict]:
        full = read_claims(_columns(score_animals), None if state_animals is None else _values(state_animals), rng, resamples, level)
        for claim, (kind, _, _) in CLAIMS.items():
            if kind == "state" and state_animals is None:
                full[claim] = {"reading": "no reading: a run condition was not met", "quantities": {}}
            elif not (conditions["enough_animals"] and (conditions["fits_met"] or kind == "state")):
                full[claim]["reading"] = "no reading: a run condition was not met"
        return full

    result = {"status": plan["status"], "level": plan["level"], "conditions": conditions,
              "claims": reading(primary, state, plan["level"]),
              "sensitivity": {
                  "without_largest_lab": {"lab_removed": largest} | reading(
                      [a for a, k in zip(primary, keep, strict=True) if k],
                      None if state is None else [a for a in state if labs[a["subject"]] != largest], plan["level"]),
                  "chronological_split": {k: v for k, v in reading(jobs["chronological"], state, plan["level"]).items()
                                          if CLAIMS[k][0] == "score"},
                  "intervals_95": reading(primary, state, 0.95)},
              "reported_without_claim": {"comparison": descriptive}}
    if state is not None:
        d1 = _values(state)["d1_block_lag"]
        b = bounds(d1, rng, resamples, 0.95)
        late = [a["block_late"] for a in state if a["block_late"] is not None]
        result["reported_without_claim"]["block_lag"] = {
            "animals": b["animals"], "mean": b["mean"][0], "interval_95": [b["low"][0], b["high"][0]],
            "late_mean": float(np.mean(late)), "development_mean": -0.024}
        result["totals"] = {key: int(sum(a[key] for a in state)) for key in ("rt_trials", "omissions", "block_changes")}
        result["nonconverged_state_animals"] = sum(not a["converged"] for a in state)
    return result


def _summarize(args: Args) -> None:
    path = args.output / "result.json"
    if path.exists():
        raise RuntimeError("Result exists; it is never replaced")
    plan = _frozen(args)
    jobs = {name: json.loads((args.output / "scores" / f"{name}.json").read_text())["animals"] for name in plan["jobs"]}
    labs = {name: lab for name, (lab, _) in _animals(args).items()}
    passed = json.loads((args.output / "check.json").read_text())["passed"]
    state = json.loads((args.output / "animals.json").read_text())["animals"] if passed else None
    result = summarize(plan, labs, jobs, state) | {"plan_sha256": _digest(args.output / "plan.json")}
    with (args.output / "subject_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["subject", "lab", "split", "sessions", "trials", *MODELS])
        for split in ("alternating", "chronological"):
            for a in jobs[split]:
                writer.writerow([a["subject"], labs[a["subject"]], split, a["sessions"], a["trials"], *(f"{a['nll'][m]:.6f}" for m in MODELS)])
    if state is not None:
        with (args.output / "subject_values.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            columns = [*SIZES, "block_late", "rt_trials", "omissions", "block_changes"]
            writer.writerow(["subject", "lab", *columns])
            for a in state:
                writer.writerow([a["subject"], labs[a["subject"]], *("" if a[c] is None else a[c] for c in columns)])
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("conditions", "claims", "sensitivity")}, indent=1))


def run(args: Args) -> None:
    """Dispatch one explicit stage."""
    {"cohort": _cohort, "plan": _plan, "download": _download, "convert": _convert, "check": _check,
     "run": _run, "summarize": _summarize}[args.stage](args)


if __name__ == "__main__":
    run(tyro.cli(Args))
