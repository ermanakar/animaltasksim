"""Fit the missing choice-only no-control comparator against saved diagnosis outputs."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

from agents.adaptive_control_config import AdaptiveControlConfig
from agents.adaptive_control_trainer import AdaptiveControlTrainer
from agents.losses import LossWeights
from eval.metrics import load_trials
from scripts.validate_architecture_prediction import digest, hard_predictions, score_rows


@dataclass(slots=True)
class Args:
    source_run: Path = Path("runs/architecture_validation_20260911/prediction")
    matched_run: Path = Path("runs/architecture_diagnosis_20260911/experiment")
    output: Path = Path("runs/matched_choice_control_20260912")
    seeds: tuple[int, ...] = (42, 123)
    checkpoints: tuple[int, ...] = (2, 4, 6)
    prediction_samples: int = 512


def relevant_source_hashes() -> dict[str, str]:
    """Hash the frozen pre-existing dependency set, excluding unrelated additions."""
    excluded = {"scripts/choice_control_comparison.py", "scripts/compare_history_explanations.py"}
    paths = [p for folder in ("agents", "envs", "eval", "animaltasksim", "scripts")
             for p in sorted(Path(folder).rglob("*.py")) if str(p) not in excluded
             and p.name != "competing_history.py"]
    return {str(p): digest(p) for p in paths}


def bootstrap_delta(before: pd.Series, after: pd.Series) -> dict:
    paired = pd.concat([before.rename("before"), after.rename("after")], axis=1).dropna()
    delta = (paired.before - paired.after).to_numpy()
    if len(delta) < 2:
        raise ValueError("Need at least two paired measurements")
    rng = np.random.default_rng(4242)
    boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(1)
    return {"mean_improvement": float(delta.mean()),
            "subject_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "subjects_improved": int((delta > 0).sum()), "subjects": len(delta)}


def main(args: Args) -> None:
    if args.checkpoints != (2, 4, 6):
        raise ValueError("Checkpoints are fixed at 2, 4, 6")
    if args.seeds != (42, 123) or args.prediction_samples != 512:
        raise ValueError("Seeds and prediction samples are fixed at (42, 123) and 512")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    if not args.matched_run.exists():
        raise RuntimeError(f"Missing matched diagnosis run: {args.matched_run}")
    torch.set_num_threads(1)
    prior = json.loads((args.source_run / "plan.json").read_text())
    reference = Path(prior["settings"]["reference"])
    identities_path = Path(prior["settings"]["identities"])
    train_log = args.source_run / "train.ndjson"
    fixed = [reference, identities_path, train_log, args.source_run / "plan.json",
             args.source_run / "predictions.csv", args.matched_run / "plan.json",
             args.matched_run / "predictions.csv", args.matched_run / "subject_scores.csv"]
    hashes = relevant_source_hashes()
    runner_hash = digest(Path(__file__))
    input_hashes = {str(p): digest(p) for p in fixed}
    matched_plan = json.loads((args.matched_run / "plan.json").read_text())
    if digest(args.matched_run / "plan.json") != (args.matched_run / "plan.sha256").read_text().strip():
        raise RuntimeError("Matched diagnosis plan hash mismatch")
    if any(digest(Path(path)) != value for path, value in matched_plan.get("input_sha256", {}).items()):
        raise RuntimeError("Matched diagnosis input hash mismatch")
    if matched_plan.get("settings", {}).get("seeds") != [42, 123] or matched_plan.get("settings", {}).get("checkpoints") != [2, 4, 6] or matched_plan.get("settings", {}).get("prediction_samples") != 512:
        raise RuntimeError("Matched diagnosis settings differ")
    if any(not Path(path).exists() or digest(Path(path)) != value
           for path, value in matched_plan["source_sha256"].items()):
        raise RuntimeError("Matched diagnosis source hashes differ")
    if digest(args.source_run / "plan.json") != (args.source_run / "plan.sha256").read_text():
        raise RuntimeError("Original plan hash mismatch")
    if digest(reference) != prior["reference_sha256"] or digest(identities_path) != prior["identity_sha256"]:
        raise RuntimeError("Original source data changed")
    identities = {s["id"]: s for s in json.loads(identities_path.read_text())["sessions"]}
    frame = load_trials(reference).sort_values(["session_id", "trial_index"]).reset_index(drop=True)
    frame["subject"] = frame.session_id.map(lambda s: identities[s]["subject"])
    test = frame[frame.subject.isin(prior["test_subjects"])].copy().reset_index(drop=True)
    training = load_trials(train_log)
    training_subjects = {identities[s]["subject"] for s in training.session_id.unique()}
    if training_subjects != set(prior["train_subjects"]) or training_subjects & set(test.subject):
        raise RuntimeError("Subject split changed or overlaps")
    if matched_plan["train_subjects"] != prior["train_subjects"] or matched_plan["development_evaluation_subjects"] != prior["test_subjects"]:
        raise RuntimeError("Matched subject sets differ")
    matched = pd.read_csv(args.matched_run / "predictions.csv")
    matched = matched[matched.model == "choice_only_full_control"].copy()
    expected_keys = set(zip(test.session_id, test.trial_index))
    for seed in args.seeds:
        for epoch in args.checkpoints:
            part = matched[(matched.seed == seed) & (matched.epoch == epoch)]
            if len(part) != len(test) or set(zip(part.session_id, part.trial_index)) != expected_keys:
                raise RuntimeError("Matched prediction coverage differs")
            saved_config = json.loads((args.matched_run / f"choice_only_full_control_{seed}" / f"epoch_{epoch}" / "config.json").read_text())
            if saved_config["loss_weights"]["rt"] != 0 or saved_config["hidden_size"] != 16 or saved_config["max_trials_per_session"] != 64:
                raise RuntimeError("Matched trained model configuration differs")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(args).items()}
    plan = {"settings": settings, "condition": "choice_only_no_control",
            "source_sha256": hashes, "runner_sha256": runner_hash, "input_sha256": input_hashes,
            "train_subjects": prior["train_subjects"], "evaluation_subjects": prior["test_subjects"],
            "schedule": [2, 4, 6], "primary_checkpoint": 6,
            "training": "Same training log, hidden16, chunk64, optimizer and seeds as matched diagnosis; RT loss weight 0.",
            "reuse": "Saved choice-only full-control and simple-history scores are reused; no matched condition is rerun.",
            "primary": "Fixed epoch6 equal-animal choice NLL, averaging seeds within each animal; positive full-minus-reduced loss favors reduced.",
            "contrasts": ["choice-only reduced vs choice-only full", "choice-only reduced vs simple history"],
            "interpretation": "Bounded development comparison; no best-epoch selection or prospective claim.",
            "limits": ["Previously inspected animals, two seeds; not a fresh test.",
                       "Timing is untrained directly and reported only as diagnostic.",
                       "Animal bootstrap omits training-population and Monte Carlo uncertainty; multiple descriptive contrasts."]}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    (args.output / "plan.sha256").write_text(digest(args.output / "plan.json"))
    baseline = pd.read_csv(args.source_run / "predictions.csv")
    baseline = baseline[baseline.model == "simple_history"].copy()
    baseline["epoch"] = 0
    scores = [baseline, matched]
    history = []
    for seed in args.seeds:
        config = AdaptiveControlConfig(reference_log=train_log, hidden_size=16, epochs=1,
                                       max_trials_per_session=64, seed=seed,
                                       loss_weights=LossWeights(choice=1, rt=0),
                                       episodes=1, trials_per_episode=1)
        config.apply_control_profile("no_control")
        trainer = AdaptiveControlTrainer(config)
        for epoch in range(1, 7):
            trainer.model.train()
            metrics = trainer.train()
            history.append({"condition": "choice_only_no_control", "seed": seed, "epoch": epoch,
                            **{key: values[0] for key, values in metrics.items()}})
            print(f"trained choice_only_no_control seed={seed} epoch={epoch} choice={metrics['epoch_choice_loss'][0]:.6f} rt={metrics['epoch_rt_loss'][0]:.6f}", flush=True)
            if epoch not in args.checkpoints:
                continue
            if relevant_source_hashes() != hashes or digest(Path(__file__)) != runner_hash or any(digest(Path(p)) != value for p, value in input_hashes.items()):
                raise RuntimeError("Source or matched inputs changed during run")
            config.output_dir = args.output / f"choice_only_no_control_{seed}" / f"epoch_{epoch}"
            config.epochs = epoch
            trainer.save(config.output_paths(), {k: [r[k] for r in history if r["seed"] == seed] for k in metrics}, {})
            config.epochs = 1
            torch.save(trainer.optimizer.state_dict(), config.output_dir / "optimizer.pt")
            ps, rts = [], []
            for sid, group in test.groupby("session_id", sort=False):
                features = trainer._session_to_arrays(group)[0]
                p, rt = hard_predictions(trainer, features, sid, args.prediction_samples)
                ps.extend(p)
                rts.extend(rt)
            scored = score_rows(test, np.array(ps), np.array(rts), "choice_only_no_control", seed)
            scored["epoch"] = epoch
            scored.to_csv(config.output_dir / "predictions.csv", index=False)
            scores.append(scored)
            print(f"scored choice_only_no_control seed={seed} epoch={epoch}", flush=True)
        pd.DataFrame(history).to_csv(args.output / "training_curves.csv", index=False)
    if relevant_source_hashes() != hashes or digest(Path(__file__)) != runner_hash or any(digest(Path(p)) != value for p, value in input_hashes.items()):
        raise RuntimeError("Frozen dependencies or inputs changed before reporting")
    if digest(args.output / "plan.json") != (args.output / "plan.sha256").read_text().strip():
        raise RuntimeError("Run plan changed")
    combined = pd.concat(scores, ignore_index=True)
    combined.to_csv(args.output / "predictions.csv", index=False)
    subject_scores = combined.groupby(["subject", "model", "seed", "epoch"])[["choice_nll", "rt_squared_error_seconds"]].mean().reset_index()
    subject_scores.to_csv(args.output / "subject_scores.csv", index=False)
    by = subject_scores.groupby(["subject", "model", "epoch"])[["choice_nll", "rt_squared_error_seconds"]].mean()
    comparisons = {}
    for name, other in [("choice_only_no_control_vs_matched_full_at_six", ("choice_only_full_control", 6)),
                        ("choice_only_no_control_vs_simple_at_six", ("simple_history", 0))]:
        before = by.xs(other, level=["model", "epoch"])
        after = by.xs(("choice_only_no_control", 6), level=["model", "epoch"])
        comparisons[name] = {metric: bootstrap_delta(before[metric], after[metric]) for metric in before.columns}
    means = by.groupby(["model", "epoch"]).mean().reset_index().to_dict("records")
    result = {"comparisons": comparisons, "equal_subject_means": means,
              "plan_sha256": digest(args.output / "plan.json"),
              "input_hashes_unchanged": all(digest(Path(p)) == value for p, value in input_hashes.items()),
              "dependency_hashes_unchanged": relevant_source_hashes() == hashes and digest(Path(__file__)) == runner_hash,
              "limits": ["Eight inspected development evaluation mice; seven have valid RT scores.",
                         "Two seeds and six epochs are bounded evidence; no best epoch was selected.",
                         "This comparison does not establish animal generalization or neural necessity."]}
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
