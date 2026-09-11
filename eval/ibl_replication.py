"""Fixed raw-data eligibility and subject-level scoring for prospective IBL replication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eval.history_audit import AuditConfig, MODELS, design_matrix, prepare_trials
from eval.schema_validator import TrialRecord

__all__ = ["ReplicationRules", "convert_source", "score_cohort"]


@dataclass(slots=True)
class ReplicationRules:
    """Rules chosen using development data only; freeze before loading the reserved cohort."""

    min_trials: int = 150
    min_full_contrast_trials: int = 20
    min_easy_accuracy: float = 0.85
    min_eligible_subjects: int = 40
    minimum_nll_gain: float = 0.0005
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 20260906


def convert_source(
    arrays: dict[str, np.ndarray], eid: str, rules: ReplicationRules,
) -> tuple[list[dict], dict[str, object]]:
    """Use ALF wheel-choice coding, retain original indices, and score preset session QC."""
    required = ("choice", "feedbackType", "contrastLeft", "contrastRight",
                "probabilityLeft", "stimOn_times", "response_times")
    n = len(arrays["choice"])
    if any(key not in arrays or len(arrays[key]) != n for key in required):
        raise ValueError("Missing or misaligned source arrays.")
    if not np.isin(arrays["choice"], [-1, 0, 1]).all() or not np.isin(arrays["feedbackType"], [-1, 1]).all():
        raise ValueError("Unknown choice/outcome code.")
    rows, previous = [], None
    dropped = 0
    for i in range(n):
        cl, cr = arrays["contrastLeft"][i], arrays["contrastRight"][i]
        if bool(np.isfinite(cl)) == bool(np.isfinite(cr)):
            raise ValueError("Unresolved stimulus side.")
        contrast = float(cr if np.isfinite(cr) else -cl)
        levels = np.array([0, 0.0625, 0.125, 0.25, 1.0])
        matches = np.isclose(abs(contrast), levels, atol=1e-8, rtol=0)
        if not matches.any():
            dropped += 1
            previous = None
            continue
        contrast = float(np.sign(contrast) * levels[np.flatnonzero(matches)[0]])
        prior = float(1 - arrays["probabilityLeft"][i])
        if not np.isclose(prior, [0.2, 0.5, 0.8], atol=1e-8, rtol=0).any():
            raise ValueError("Unknown block prior.")
        raw_choice = arrays["choice"][i]
        action = 0 if raw_choice == -1 else (1 if raw_choice == 1 else 2)
        correct = bool(arrays["feedbackType"][i] == 1)
        if action == 2 and correct:
            raise ValueError("Rewarded no-go is inconsistent.")
        if contrast and action != 2 and (((action == 0) == (contrast > 0)) != correct):
            raise ValueError("Choice-side/outcome inconsistency.")
        elapsed = float((arrays["response_times"][i] - arrays["stimOn_times"][i]) * 1000)
        rt = elapsed if action != 2 and np.isfinite(elapsed) and elapsed > 0 else None
        row = {"task": "IBL2AFC", "session_id": eid, "trial_index": i,
               "stimulus": {"contrast": contrast}, "block_prior": {"p_right": prior},
               "action": action, "correct": correct, "reward": float(correct), "rt_ms": rt,
               "phase_times": {}, "prev": previous, "seed": 0,
               "agent": {"name": "reference_mouse", "version": "ibl_replication_v1"}}
        TrialRecord.model_validate(row)
        rows.append(row)
        previous = {key: row[key] for key in ("action", "reward", "correct")}
    easy = [r["correct"] for r in rows if abs(r["stimulus"]["contrast"]) == 1]
    accuracy = float(np.mean(easy)) if easy else None
    eligible = (len(rows) >= rules.min_trials and len(easy) >= rules.min_full_contrast_trials
                and accuracy is not None and accuracy >= rules.min_easy_accuracy)
    return rows, {"session_id": eid, "raw_trials": n, "retained_trials": len(rows),
                  "dropped_contrast": dropped, "easy_trials": len(easy), "easy_accuracy": accuracy,
                  "omissions": sum(r["action"] == 2 for r in rows), "eligible": bool(eligible)}


def score_cohort(
    df: pd.DataFrame, models: dict, session_subjects: dict[str, str],
    analysis: AuditConfig, rules: ReplicationRules,
) -> dict[str, object]:
    """Score fixed development fits; bootstrap test subjects without refitting or tuning."""
    data = prepare_trials(df, analysis)
    if any(sid not in session_subjects for sid in data.session_id):
        raise ValueError("Missing test subject identity.")
    data["subject"] = data.session_id.map(session_subjects)
    losses = {}
    for model in MODELS:
        x, names = design_matrix(data, model)
        if names != models[model]["features"]:
            raise ValueError("Frozen feature order changed.")
        logits = x @ np.array(models[model]["coefficients"])
        losses[model] = np.logaddexp(0, logits) - data.choice_right.to_numpy() * logits
    per_subject = []
    for subject, group in data.groupby("subject", sort=True):
        per_subject.append({"subject": str(subject), "n_trials": len(group),
                            "nll": {model: float(loss[group.index].mean()) for model, loss in losses.items()}})
    delta = np.array([r["nll"]["outcome_history"] - r["nll"]["evidence_history"] for r in per_subject])
    draws = np.random.default_rng(rules.bootstrap_seed).choice(delta, size=(rules.bootstrap_samples, len(delta)), replace=True).mean(axis=1)
    lower, upper = map(float, np.quantile(draws, [0.025, 0.975]))
    sufficient = len(per_subject) >= rules.min_eligible_subjects
    success = sufficient and float(delta.mean()) >= rules.minimum_nll_gain and lower > 0
    return {
        "status": "prospective_internal_replication_pass" if success else (
            "insufficient_eligible_subjects" if not sufficient else "prospective_internal_replication_did_not_pass"),
        "n_subjects": len(per_subject), "eligible_transitions": len(data),
        "primary_equal_subject_nll_gain": float(delta.mean()), "subject_bootstrap_95_interval": [lower, upper],
        "subjects_improved": int(np.sum(delta > 0)), "criterion_met": success,
        "pooled_nll": {model: float(loss.mean()) for model, loss in losses.items()},
        "per_subject": per_subject,
        "limits": ["Internal freeze, not an externally preregistered study.",
                   "Subject novelty is relative to inventoried local references; earlier unrecorded exposure cannot be excluded.",
                   "Bootstrap conditions on these fixed trained models and samples test subjects, not training datasets or labs.",
                   "This tests a narrow predictive interaction, not mechanism, anatomy, RT fit, or scientific novelty."],
    }
