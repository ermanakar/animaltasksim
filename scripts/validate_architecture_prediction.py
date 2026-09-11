"""Frozen exploratory subject-disjoint prediction gate for the repaired architecture."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from scipy.optimize import minimize
from scipy.special import expit

from agents.adaptive_control_config import AdaptiveControlConfig
from agents.adaptive_control_trainer import AdaptiveControlTrainer
from agents.losses import LossWeights
from eval.metrics import load_trials


@dataclass(slots=True)
class Args:
    output: Path = Path("runs/architecture_validation_20260911/prediction")
    reference: Path = Path("runs/ibl_source_reconciled/reference.ndjson")
    identities: Path = Path("runs/ibl_source_audit/source_manifest.json")
    train_subjects: int = 12
    test_subjects: int = 8
    split_seed: int = 20260911
    seeds: tuple[int, ...] = (42, 123)
    epochs: int = 2
    prediction_samples: int = 512


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(p): digest(p) for folder in ("agents", "envs", "eval", "animaltasksim", "scripts")
            for p in sorted(Path(folder).rglob("*.py"))}


def split_subjects(subjects: list[str], train_n: int, test_n: int, seed: int) -> tuple[list[str], list[str]]:
    if train_n < 1 or test_n < 2 or train_n + test_n > len(set(subjects)):
        raise ValueError("Need disjoint nonempty training and at least two test subjects")
    ordered = sorted(set(subjects), key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    return ordered[:train_n], ordered[train_n:train_n + test_n]


def design(features: np.ndarray) -> np.ndarray:
    """Same observable inputs plus two prespecified one-step interactions; no block oracle."""
    return np.column_stack((np.ones(len(features)), features,
                            features[:, 3] * features[:, 4],
                            features[:, 3] * (1 - np.abs(features[:, 0]))))


def fit_baseline(features: np.ndarray, choice: np.ndarray, choice_mask: np.ndarray,
                 rt: np.ndarray, rt_mask: np.ndarray) -> dict:
    x = design(features).astype(float)
    center, scale = x.mean(0), x.std(0)
    center[0], scale[0] = 0, 1
    scale[scale < 1e-8] = 1
    x = (x - center) / scale
    valid = choice_mask > 0
    xx, yy = x[valid], choice[valid]
    penalty = np.ones(x.shape[1]) * .001
    penalty[0] = 0

    def objective(beta):
        z = xx @ beta
        return (np.mean(np.logaddexp(0, z) - yy * z) + .5 * np.sum(penalty * beta**2),
                xx.T @ (expit(z) - yy) / len(yy) + penalty * beta)

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "gtol": 1e-9})
    if not result.success:
        raise RuntimeError(f"Baseline fit failed: {result.message}")
    valid_rt = rt_mask > 0
    xr, yr = x[valid_rt], rt[valid_rt] / 1000
    rt_beta = np.linalg.solve(xr.T @ xr / len(yr) + np.diag(penalty + 1e-10), xr.T @ yr / len(yr))
    return {"center": center.tolist(), "scale": scale.tolist(), "choice_beta": result.x.tolist(),
            "rt_beta": rt_beta.tolist(), "ridge": .001, "converged": bool(result.success)}


def baseline_predict(fit: dict, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (design(features) - fit["center"]) / fit["scale"]
    return expit(x @ fit["choice_beta"]), np.clip(x @ fit["rt_beta"], .05, 3.0) * 1000


@torch.no_grad()
def hard_predictions(trainer: AdaptiveControlTrainer, features: np.ndarray, session_id: str,
                     samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced history; hard Euler crossing with lapse integrated analytically."""
    model = trainer.model
    model.eval()
    h, c = model.init_state()
    plastic, trace, value, gate, change = model.init_plastic_state()
    p_right, rt = [], []
    steps = np.arange(1, 301)
    for index, row in enumerate(features):
        x = torch.from_numpy(row).reshape(1, -1)
        plastic, trace, _, change = model.update_plastic_history(
            plastic_state=plastic, eligibility_trace=trace, prev_action=x[:, 3],
            prev_reward=x[:, 4], prev_value_prediction=value,
            prev_history_gate=gate, change_evidence=change)
        out, (h, c) = model(x, (h, c), plastic_state=plastic)
        value, gate = out["critic_value"].reshape(1, 1), out["perceptual_gate"].reshape(1, 1)
        bound, noise = out["bound"].item(), out["noise"].item()
        stay = out["stay_tendency"].item()
        start = np.clip(out["bias"].item() + stay * model.effective_history_bias_scale.item() * bound * row[3],
                        -bound * (1 - 1e-6), bound * (1 - 1e-6))
        drift = out["drift_gain"].item() * row[0] + stay * model.history_drift_scale.item() * row[3] * (1 - min(abs(row[0]), 1))
        seed = int.from_bytes(hashlib.sha256(f"prediction:{session_id}:{index}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        trajectory = start + np.cumsum(drift * .01 + noise * .1 * rng.standard_normal((samples, 300)), axis=1)
        crossing = np.abs(trajectory) >= bound
        first = np.where(crossing.any(1), crossing.argmax(1), 299)
        decision = trajectory[np.arange(samples), first] > 0
        ndt = out["non_decision_ms"].item()
        response_rt = np.clip(np.floor((first + 1) + ndt / 10), 5, 300) * 10
        lapse_rt = np.clip(np.floor(steps[4:] + ndt / 10), 5, 300).mean() * 10
        p_right.append(.95 * decision.mean() + .05 * .5)
        rt.append(.95 * response_rt.mean() + .05 * lapse_rt)
    return np.array(p_right), np.array(rt)


def score_rows(frame: pd.DataFrame, probabilities: np.ndarray, rt_prediction: np.ndarray,
               model: str, seed: int) -> pd.DataFrame:
    valid = frame.action.isin(["left", "right"]).to_numpy()
    y = (frame.action == "right").to_numpy().astype(float)
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    actual_rt = frame.rt_ms.to_numpy(dtype=float)
    rt_valid = valid & np.isfinite(actual_rt) & (actual_rt > 0)
    return pd.DataFrame({"subject": frame.subject.to_numpy(), "session_id": frame.session_id.to_numpy(),
                         "trial_index": frame.trial_index.to_numpy(), "model": model, "seed": seed,
                         "choice_nll": np.where(valid, -(y * np.log(p) + (1-y)*np.log1p(-p)), np.nan),
                         "rt_squared_error_seconds": np.where(rt_valid, ((actual_rt-rt_prediction)/1000)**2, np.nan),
                         "predicted_p_right": p, "predicted_rt_ms": rt_prediction,
                         "rt_outside_window": rt_valid & ((actual_rt < 50) | (actual_rt > 3000)),
                         "omission": ~valid})


def paired_summary(scores: pd.DataFrame) -> dict:
    # Average training seeds within each subject; subjects are the sampling unit.
    subject = scores.groupby(["subject", "model"])[["choice_nll", "rt_squared_error_seconds"]].mean()
    output = {}
    for candidate, baseline in [("full_control", "simple_history"), ("full_control", "no_control")]:
        key = f"{candidate}_vs_{baseline}"
        result = {}
        for metric in subject.columns:
            delta = (subject.xs(baseline, level="model")[metric] - subject.xs(candidate, level="model")[metric]).dropna().to_numpy()
            rng = np.random.default_rng(4242)
            boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(1)
            result[metric] = {"mean_improvement": float(delta.mean()),
                              "subject_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
                              "subjects_improved": int((delta > 0).sum()), "subjects": len(delta)}
        result["exploratory_gate_pass"] = all(r["subject_bootstrap_95"][0] > 0 for r in result.values())
        output[key] = result
    return output


def main(args: Args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a fresh output directory; results cannot be overwritten")
    torch.set_num_threads(1)
    identities = json.loads(args.identities.read_text())["sessions"]
    identity = {s["id"]: s for s in identities}
    frame = load_trials(args.reference).sort_values(["session_id", "trial_index"]).reset_index(drop=True)
    if not set(frame.session_id) <= set(identity):
        raise ValueError("Missing subject identities")
    frame["subject"] = frame.session_id.map(lambda sid: identity[sid]["subject"])
    train_ids, test_ids = split_subjects(frame.subject.unique().tolist(), args.train_subjects, args.test_subjects, args.split_seed)
    hashes = source_hashes()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"settings": {**asdict(args), "output": str(args.output), "reference": str(args.reference), "identities": str(args.identities)},
            "train_subjects": train_ids, "test_subjects": test_ids,
            "source_sha256": hashes, "reference_sha256": digest(args.reference), "identity_sha256": digest(args.identities),
            "models": ["simple_history", "no_control", "full_control"],
            "training": "Two epochs, seeds42/123, same full training sessions, hidden16, chunk64, 32 paths; no validation tuning.",
            "primary": "Equal-subject held-out choice NLL improvement; secondary RT MSE in seconds squared, all valid RTs retained.",
            "gate": "Full model needs positive subject-bootstrap lower limits for both choice and RT improvements against both alternatives.",
            "limits": ["Previously inspected development animals: exploratory, not a fresh prospective replication.",
                       "Teacher-forced one-step predictions use observed previous feedback, not current outcomes or block priors.",
                       "Choice NLL is finite-Monte-Carlo approximate; RT score tests mean prediction, not full joint likelihood.",
                       "Two epochs is a bounded viability gate, not equal convergence or exhaustive tuning.",
                       "Single split, eight evaluation subjects and two seeds cannot establish general robustness.",
                       "Reported intervals omit Monte Carlo and training-population uncertainty; no novelty or neural claim."]}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    (args.output / "plan.sha256").write_text(digest(args.output / "plan.json"))
    train_frame = frame[frame.subject.isin(train_ids)].copy()
    test_frame = frame[frame.subject.isin(test_ids)].copy().reset_index(drop=True)
    train_session_ids = set(train_frame.session_id)
    with args.reference.open() as source, (args.output / "train.ndjson").open("w") as dest:
        for line in source:
            if json.loads(line)["session_id"] in train_session_ids:
                dest.write(line)
    config = AdaptiveControlConfig(reference_log=args.output / "train.ndjson", hidden_size=16,
                                   epochs=args.epochs, max_trials_per_session=64,
                                   loss_weights=LossWeights(choice=1, rt=1), episodes=1, trials_per_episode=1)
    extractor = object.__new__(AdaptiveControlTrainer)
    extractor.config = config
    def arrays(df):
        parts = [extractor._session_to_arrays(g) for _, g in df.groupby("session_id", sort=False)]
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(6))
    train_arrays = arrays(train_frame)
    baseline = fit_baseline(*train_arrays[:5])
    (args.output / "baseline.json").write_text(json.dumps(baseline, indent=2))
    test_arrays = arrays(test_frame)
    p, rt = baseline_predict(baseline, test_arrays[0])
    scores = [score_rows(test_frame, p, rt, "simple_history", -1)]
    for seed in args.seeds:
        for profile in ("no_control", "full_control"):
            if source_hashes() != hashes:
                raise RuntimeError("Code changed after plan freeze")
            config.seed, config.output_dir = seed, args.output / f"{profile}_{seed}"
            config.apply_control_profile(profile)
            trainer = AdaptiveControlTrainer(config)
            metrics = trainer.train()
            trainer.save(config.output_paths(), metrics, {})
            print(f"trained {profile} seed={seed}", flush=True)
            ps, rts = [], []
            for sid, group in test_frame.groupby("session_id", sort=False):
                features = extractor._session_to_arrays(group)[0]
                pp, rr = hard_predictions(trainer, features, sid, args.prediction_samples)
                ps.extend(pp)
                rts.extend(rr)
            scored = score_rows(test_frame, np.array(ps), np.array(rts), profile, seed)
            scored.to_csv(config.output_dir / "heldout_predictions.csv", index=False)
            scores.append(scored)
            print(f"scored {profile} seed={seed}", flush=True)
    combined = pd.concat(scores, ignore_index=True)
    combined.to_csv(args.output / "predictions.csv", index=False)
    combined.groupby(["subject", "model", "seed"])[["choice_nll", "rt_squared_error_seconds"]].mean().to_csv(args.output / "subject_scores.csv")
    result = {"comparisons": paired_summary(combined), "train_trials": len(train_frame), "test_trials": len(test_frame),
              "train_sessions": train_frame.session_id.nunique(), "test_sessions": test_frame.session_id.nunique(),
              "test_omissions": int(scores[0].omission.sum()), "test_rt_outside_window": int(scores[0].rt_outside_window.sum()),
              "subject_mean_scores": combined.groupby(["subject", "model"])[["choice_nll", "rt_squared_error_seconds"]].mean().groupby("model").mean().to_dict("index"),
              "plan_sha256": digest(args.output / "plan.json"), "interpretation": plan["limits"]}
    if source_hashes() != hashes or digest(args.reference) != plan["reference_sha256"]:
        raise RuntimeError("Inputs changed during experiment")
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
