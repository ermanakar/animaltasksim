"""Prespecified development diagnosis of training duration and RT-loss interference."""
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
from scripts.validate_architecture_prediction import (
    digest, hard_predictions, score_rows, source_hashes,
)
from eval.metrics import load_trials


CONDITIONS = {"raw_no_control": ("no_control", 1.0),
              "raw_full_control": ("full_control", 1.0),
              "choice_only_full_control": ("full_control", 0.0)}


@dataclass(slots=True)
class Args:
    source_run: Path = Path("runs/architecture_validation_20260911/prediction")
    output: Path = Path("runs/architecture_diagnosis_20260911/experiment")
    seeds: tuple[int, ...] = (42, 123)
    checkpoints: tuple[int, ...] = (2, 4, 6)
    prediction_samples: int = 512


def bootstrap_delta(before: pd.Series, after: pd.Series) -> dict:
    paired = pd.concat([before.rename("before"), after.rename("after")], axis=1).dropna()
    delta = (paired.before - paired.after).to_numpy()
    if len(delta) < 2:
        raise ValueError("Need at least two animals with paired measurements")
    rng = np.random.default_rng(4242)
    boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(1)
    return {"mean_improvement": float(delta.mean()),
            "subject_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "subjects_improved": int((delta > 0).sum()), "subjects": len(delta)}


def summarize(scores: pd.DataFrame) -> dict:
    by_subject = scores.groupby(["subject", "model", "epoch"])[["choice_nll", "rt_squared_error_seconds"]].mean()
    comparisons = {}
    for name, earlier, later in [
        ("longer_training_full", ("raw_full_control", 2), ("raw_full_control", 6)),
        ("longer_training_reduced", ("raw_no_control", 2), ("raw_no_control", 6)),
        ("rt_loss_removal_at_six", ("raw_full_control", 6), ("choice_only_full_control", 6)),
        ("controller_at_six", ("raw_no_control", 6), ("raw_full_control", 6)),
        ("full_vs_simple_at_six", ("simple_history", 0), ("raw_full_control", 6)),
        ("choice_only_vs_simple_at_six", ("simple_history", 0), ("choice_only_full_control", 6)),
    ]:
        before = by_subject.xs(earlier, level=["model", "epoch"])
        after = by_subject.xs(later, level=["model", "epoch"])
        comparisons[name] = {metric: bootstrap_delta(before[metric], after[metric]) for metric in before.columns}
    means = by_subject.groupby(["model", "epoch"]).mean().reset_index().to_dict("records")
    return {"comparisons": comparisons, "equal_subject_means": means}


def main(args: Args) -> None:
    if args.checkpoints != (2, 4, 6):
        raise ValueError("This diagnostic fixes checkpoints at2,4,6; use a new protocol for other schedules")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    torch.set_num_threads(1)
    previous = json.loads((args.source_run / "plan.json").read_text())
    if digest(args.source_run / "plan.json") != (args.source_run / "plan.sha256").read_text():
        raise RuntimeError("Original plan hash mismatch")
    reference = Path(previous["settings"]["reference"])
    identities_path = Path(previous["settings"]["identities"])
    if digest(reference) != previous["reference_sha256"] or digest(identities_path) != previous["identity_sha256"]:
        raise RuntimeError("Original source data changed")
    identities = {s["id"]: s for s in json.loads(identities_path.read_text())["sessions"]}
    frame = load_trials(reference).sort_values(["session_id", "trial_index"]).reset_index(drop=True)
    frame["subject"] = frame.session_id.map(lambda s: identities[s]["subject"])
    test = frame[frame.subject.isin(previous["test_subjects"])].copy().reset_index(drop=True)
    train_log = args.source_run / "train.ndjson"
    training = load_trials(train_log)
    training_subjects = {identities[s]["subject"] for s in training.session_id.unique()}
    if training_subjects != set(previous["train_subjects"]) or training_subjects & set(test.subject):
        raise RuntimeError("Subject split changed or overlaps")
    hashes = source_hashes()
    fixed_inputs = {str(p): digest(p) for p in [reference, identities_path, train_log,
                   args.source_run / "plan.json", args.source_run / "predictions.csv"]}
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"settings": {**asdict(args), "source_run": str(args.source_run), "output": str(args.output)},
            "conditions": CONDITIONS, "source_sha256": hashes, "input_sha256": fixed_inputs,
            "train_subjects": previous["train_subjects"], "development_evaluation_subjects": previous["test_subjects"],
            "primary_checkpoint": 6, "schedule": [2,4,6],
            "interpretation": "Post-result development diagnosis, not fresh confirmation; no choice of best epoch after scoring.",
            "contrasts": ["raw full2 vs6", "raw reduced2 vs6", "raw full6 vs choice-only full6", "raw reduced6 vs full6", "simple vs full6 and choice-only6"],
            "controls": "Same initialization seeds, reference, update count, optimizer and controller regularization; only RT-loss weight differs for choice-only.",
            "rt_policy": "Choice-only leaves RT untrained by its direct loss. Score raw RT unchanged, retain missingness and out-of-window limits; do not claim joint validation.",
            "decision": "Report epoch6 and all trajectories; lower animal-bootstrap95% >0 is descriptive support, not multiplicity-adjusted confirmation. If curves still move, convergence remains unresolved.",
            "limits": ["Eight already-inspected evaluation mice; seven have RTs.", "Two seeds and six epochs are bounded optimization evidence.", "Finite-Monte-Carlo choice scoring; RT MSE is not a full distribution score.", "No correction for multiple exploratory contrasts."]}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    (args.output / "plan.sha256").write_text(digest(args.output / "plan.json"))
    baseline = pd.read_csv(args.source_run / "predictions.csv")
    baseline = baseline[baseline.model == "simple_history"].copy()
    baseline["epoch"] = 0
    scores = [baseline]
    history = []
    for seed in args.seeds:
        for name, (profile, rt_weight) in CONDITIONS.items():
            config = AdaptiveControlConfig(reference_log=train_log, hidden_size=16, epochs=1,
                                           max_trials_per_session=64, seed=seed,
                                           loss_weights=LossWeights(choice=1, rt=rt_weight),
                                           episodes=1, trials_per_episode=1)
            config.apply_control_profile(profile)
            trainer = AdaptiveControlTrainer(config)
            for epoch in range(1, 7):
                trainer.model.train()
                metrics = trainer.train()
                history.append({"condition": name, "seed": seed, "epoch": epoch,
                                **{key: values[0] for key, values in metrics.items()}})
                print(f"trained {name} seed={seed} epoch={epoch} choice={metrics['epoch_choice_loss'][0]:.6f} rt={metrics['epoch_rt_loss'][0]:.6f}", flush=True)
                if epoch not in args.checkpoints:
                    continue
                if source_hashes() != hashes:
                    raise RuntimeError("Source changed during run")
                config.output_dir = args.output / f"{name}_{seed}" / f"epoch_{epoch}"
                config.epochs = epoch
                trainer.save(config.output_paths(), {k:[r[k] for r in history if r['condition']==name and r['seed']==seed] for k in metrics}, {})
                config.epochs = 1
                torch.save(trainer.optimizer.state_dict(), config.output_dir / "optimizer.pt")
                if epoch == 2 and rt_weight == 1:
                    old = torch.load(args.source_run / f"{profile}_{seed}" / "model.pt", weights_only=True)
                    if any(not torch.equal(value, old[key]) for key, value in trainer.model.state_dict().items()):
                        raise RuntimeError("Two-epoch checkpoint differs from original; comparison integrity failed")
                ps, rts = [], []
                for sid, group in test.groupby("session_id", sort=False):
                    features = trainer._session_to_arrays(group)[0]
                    p, rt = hard_predictions(trainer, features, sid, args.prediction_samples)
                    ps.extend(p)
                    rts.extend(rt)
                scored = score_rows(test, np.array(ps), np.array(rts), name, seed)
                scored["epoch"] = epoch
                scored.to_csv(config.output_dir / "predictions.csv", index=False)
                scores.append(scored)
                print(f"scored {name} seed={seed} epoch={epoch}", flush=True)
            pd.DataFrame(history).to_csv(args.output / "training_curves.csv", index=False)
    combined = pd.concat(scores, ignore_index=True)
    combined.to_csv(args.output / "predictions.csv", index=False)
    combined.groupby(["subject", "model", "seed", "epoch"])[["choice_nll", "rt_squared_error_seconds"]].mean().to_csv(args.output / "subject_scores.csv")
    result = summarize(combined)
    result['plan_sha256'] = digest(args.output / "plan.json")
    result['two_epoch_raw_checkpoints_match_original'] = True
    result['limits'] = plan['limits']
    if source_hashes() != hashes or any(digest(Path(p)) != value for p,value in fixed_inputs.items()):
        raise RuntimeError("Inputs changed during experiment")
    (args.output / "results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
