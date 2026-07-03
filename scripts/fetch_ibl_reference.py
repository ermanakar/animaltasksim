#!/usr/bin/env python
"""Fetch multiple IBL biased-blocks sessions into a schema-valid reference log.

This is a one-time data-acquisition tool, not a runtime dependency: it requires
``ONE-api`` (``pip install ONE-api``) and network access, and it writes a
``.ndjson`` file that the rest of the project consumes offline. Install ONE-api
into a throwaway environment rather than the project env — it is intentionally
kept out of ``pyproject.toml``.

It expands the animal reference set beyond the current 10-session
``data/ibl/reference.ndjson`` by pulling ``biasedChoiceWorld`` sessions from the
IBL public server (OpenAlyx, anonymous access) and converting each trial into
the project's ``TrialRecord`` schema, matching the conventions of the existing
reference (``task="IBL2AFC"``, signed ``contrast``, integer actions, per-session
``prev``).

Two correctness safeguards, deliberately stronger than the legacy
``scripts/ibl_to_ndjson.py``:

1. **Action is derived convention-agnostically.** For non-zero contrast the
   chosen side follows unambiguously from the stimulus side and
   ``feedbackType`` (correct → chose the stimulus side; error → chose the other
   side). The notoriously confusing IBL ``choice`` sign is therefore *not*
   trusted; it is only needed for zero-contrast trials (no correct side), and
   for those the sign is *auto-calibrated* from the non-zero trials and its
   agreement with the legacy ``-1=left/+1=right`` assumption is reported.
2. **Reaction time matches the existing reference.** The default
   ``firstMovement`` source (``firstMovement_times - stimOn_times``) reproduces
   the ~378 ms median of ``reference.ndjson``. ``response`` uses
   ``response_times - stimOn_times`` (a movement-completion measure, ~1 s+).
   Choosing the wrong one silently shifts every chronometric target, so it is an
   explicit, logged choice.

The output is written alongside a manifest of the exact session EIDs so the pull
is reproducible. By design it does **not** overwrite ``reference.ndjson``: keep
the historical baseline and compare targets with
``scripts/compute_reference_targets.py`` before adopting anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

# Biased-blocks contrast set (|contrast|). 0.5 is intentionally excluded — it is
# not part of the IBL biased-blocks protocol and its presence inflated the
# psychometric slope in the past (see FINDINGS "protocol fidelity").
BIASED_BLOCK_ABS_CONTRASTS = (0.0, 0.0625, 0.125, 0.25, 1.0)
_CONTRAST_TOL = 1e-4

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_NO_OP = 2

# Reference logs use the integer convention of eval.metrics.load_trials and the
# existing data/ibl/reference.ndjson: 0 = right, 1 = left, 2 = no-op. This is the
# INVERSE of the env's ACTION_LEFT=0/ACTION_RIGHT=1 constants, so actions are
# remapped to this convention at emission time. Emitting the env convention makes
# load_trials relabel every right choice as left and inverts the psychometric.
_REFERENCE_ACTION = {ACTION_RIGHT: 0, ACTION_LEFT: 1, ACTION_NO_OP: 2}


@dataclass(slots=True)
class Args:
    """Fetch IBL biased-blocks sessions into an expanded reference log."""

    out: Path = Path("data/ibl/reference_expanded.ndjson")
    manifest: Path | None = None  # default: <out>.manifest.json
    max_sessions: int = 40
    task_protocol: str = "biasedChoiceWorld"
    rt_source: Literal["firstMovement", "response"] = "firstMovement"
    min_trials: int = 150  # drop short/aborted sessions
    min_easy_accuracy: float = 0.85  # trained-performance QC on |contrast|=1.0 trials
    min_full_contrast_trials: int = 20  # need enough easy trials to score QC reliably
    base_url: str = "https://openalyx.internationalbrainlab.org"
    seed: int = 0  # deterministic session ordering only; not a model seed


def _is_nan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _get(trials: dict, key: str, index: int) -> object:
    arr = trials.get(key)
    if arr is None:
        return None
    try:
        return arr[index]
    except (IndexError, TypeError):
        return None


def signed_contrast(trials: dict, index: int) -> float | None:
    """Return contrastRight (positive) or -contrastLeft (negative); 0.0 kept."""
    cl = _get(trials, "contrastLeft", index)
    cr = _get(trials, "contrastRight", index)
    if cr is not None and not _is_nan(cr):
        return float(cr)
    if cl is not None and not _is_nan(cl):
        return -float(cl)
    return None


def p_right(trials: dict, index: int) -> float | None:
    """Block prior expressed as P(reward on right) = 1 - probabilityLeft."""
    p_left = _get(trials, "probabilityLeft", index)
    if p_left is None or _is_nan(p_left):
        return None
    return float(1.0 - float(p_left))


def reaction_time_ms(trials: dict, index: int, source: str) -> float | None:
    """Reaction time in ms for the selected source, or None if unavailable."""
    stim_on = _get(trials, "stimOn_times", index)
    end_key = "firstMovement_times" if source == "firstMovement" else "response_times"
    end = _get(trials, end_key, index)
    if stim_on is None or end is None or _is_nan(stim_on) or _is_nan(end):
        return None
    rt = (float(end) - float(stim_on)) * 1000.0
    if not math.isfinite(rt) or rt <= 0.0:
        return None
    return rt


def in_biased_block_contrast_set(signed: float | None) -> bool:
    """True if |signed contrast| is one of the biased-blocks levels."""
    if signed is None:
        return False
    abs_c = abs(signed)
    return any(abs(abs_c - level) <= _CONTRAST_TOL for level in BIASED_BLOCK_ABS_CONTRASTS)


def derive_action(
    signed: float | None,
    feedback_type: object,
    choice: object,
    right_choice_value: float | None,
) -> int | None:
    """Map a trial to an integer action without trusting the IBL choice sign.

    For non-zero contrast the chosen side follows from the stimulus side and
    correctness. For zero-contrast trials (no correct side) the calibrated
    ``choice`` sign is used. Returns None if the trial cannot be classified.
    """
    if _is_nan(feedback_type) or feedback_type is None:
        return None
    ft = int(feedback_type)
    if signed is not None and abs(signed) > _CONTRAST_TOL:
        stim_is_right = signed > 0.0
        if ft == 1:  # correct → chose the stimulus side
            chose_right = stim_is_right
        elif ft == -1:  # error → chose the other side
            chose_right = not stim_is_right
        else:  # no-go / no feedback
            return ACTION_NO_OP
        return ACTION_RIGHT if chose_right else ACTION_LEFT
    # Zero-contrast: fall back to the (auto-calibrated) choice sign.
    if choice is None or _is_nan(choice) or float(choice) == 0.0:
        return ACTION_NO_OP
    if right_choice_value is None:
        return None
    return ACTION_RIGHT if float(choice) == right_choice_value else ACTION_LEFT


def calibrate_choice_sign(trials: dict, n: int) -> tuple[float | None, float, int]:
    """Learn which raw ``choice`` value means "right" from non-zero-contrast trials.

    Returns ``(right_choice_value, agreement, n_used)`` where ``agreement`` is the
    fraction of informative trials consistent with the mapping
    "``choice == right_choice_value`` iff the feedback-derived side is right".
    A value near 1.0 means the sign is unambiguous; near 0.5 means the ``choice``
    column carries no reliable side information.
    """
    # counts[choice_value] = [chose_right_count, chose_left_count]
    counts: dict[float, list[int]] = {}
    n_used = 0
    for i in range(n):
        signed = signed_contrast(trials, i)
        ft = _get(trials, "feedbackType", i)
        ch = _get(trials, "choice", i)
        if signed is None or abs(signed) <= _CONTRAST_TOL:
            continue
        if ft is None or _is_nan(ft) or int(ft) not in (1, -1):
            continue
        if ch is None or _is_nan(ch) or float(ch) == 0.0:
            continue
        stim_is_right = signed > 0.0
        chose_right = stim_is_right if int(ft) == 1 else not stim_is_right
        bucket = counts.setdefault(float(ch), [0, 0])
        bucket[0 if chose_right else 1] += 1
        n_used += 1
    if not counts or n_used == 0:
        return None, 0.0, n_used
    # The "right" choice value is the one most associated with rightward choices.
    right_value = max(counts, key=lambda k: counts[k][0] / max(1, sum(counts[k])))
    # Agreement: trials whose choice value agrees with the derived side.
    consistent = 0
    for value, (chose_right_count, chose_left_count) in counts.items():
        consistent += chose_right_count if value == right_value else chose_left_count
    return right_value, consistent / n_used, n_used


def _trials_len(trials: dict) -> int:
    lengths = []
    for key in ("choice", "feedbackType", "contrastLeft", "contrastRight"):
        arr = trials.get(key)
        if arr is not None:
            try:
                lengths.append(len(arr))
            except TypeError:
                pass
    return max(lengths) if lengths else 0


def session_to_records(
    trials: dict,
    session_id: str,
    rt_source: str,
) -> tuple[list[dict], dict]:
    """Convert one session's trials object into schema records + a summary."""
    n = _trials_len(trials)
    right_value, agreement, n_calib = calibrate_choice_sign(trials, n)
    records: list[dict] = []
    prev: dict | None = None
    kept = dropped_contrast = dropped_unclassified = 0
    for i in range(n):
        signed = signed_contrast(trials, i)
        if not in_biased_block_contrast_set(signed):
            dropped_contrast += 1
            continue
        action = derive_action(
            signed, _get(trials, "feedbackType", i), _get(trials, "choice", i), right_value
        )
        if action is None:
            dropped_unclassified += 1
            continue
        ft = _get(trials, "feedbackType", i)
        correct = bool(ft is not None and not _is_nan(ft) and int(ft) == 1)
        record = {
            "task": "IBL2AFC",
            "session_id": session_id,
            "trial_index": len(records),
            "stimulus": {"contrast": float(signed)},
            "block_prior": {"p_right": p_right(trials, i)},
            "action": _REFERENCE_ACTION[action],
            "correct": correct,
            "reward": 1.0 if correct else 0.0,
            "rt_ms": reaction_time_ms(trials, i, rt_source),
            "phase_times": {},
            "prev": prev,
            "seed": 0,
            "agent": {"name": "reference_mouse", "version": "ibl_public_one_expanded"},
        }
        records.append(record)
        prev = {"action": record["action"], "reward": record["reward"], "correct": record["correct"]}
        kept += 1
    # Trained-performance QC: accuracy on full-contrast (|contrast|=1.0) trials.
    # Untrained/early sessions sit near chance here and must be excluded so the
    # reference reflects mice that could actually see the stimulus.
    full_contrast = [r["correct"] for r in records if abs(r["stimulus"]["contrast"]) == 1.0]
    n_full = len(full_contrast)
    easy_accuracy = (sum(full_contrast) / n_full) if n_full > 0 else None
    summary = {
        "session_id": session_id,
        "kept": kept,
        "dropped_off_protocol_contrast": dropped_contrast,
        "dropped_unclassified": dropped_unclassified,
        "easy_full_contrast_accuracy": round(easy_accuracy, 4) if easy_accuracy is not None else None,
        "n_full_contrast": n_full,
        "choice_sign_right_value": right_value,
        "choice_sign_agreement": round(agreement, 4),
        "choice_sign_calibration_trials": n_calib,
        "choice_sign_matches_legacy_assumption": (right_value == 1.0) if right_value is not None else None,
    }
    return records, summary


def fetch(args: Args) -> None:
    """Query sessions, convert them, and write the expanded reference + manifest."""
    try:
        from one.api import ONE
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "ONE-api is required. Install it in a throwaway env: pip install ONE-api"
        ) from exc

    # Public OpenAlyx is read-only but still requires the shared anonymous
    # credentials (user "intbrainlab" / password "international"). Passing them
    # explicitly avoids relying on a previously cached token from ONE.setup().
    one = ONE(
        base_url=args.base_url,
        username="intbrainlab",
        password="international",
        silent=True,
    )
    print(f"Searching for '{args.task_protocol}' sessions on {args.base_url} ...")
    # task_protocol is filtered against the remote Alyx REST API, not the local
    # cache table, so force a remote query.
    eids = list(one.search(task_protocol=args.task_protocol, query_type="remote"))
    # Alyx returns sessions in a fixed order that front-loads early/low-quality
    # sessions and can group by lab. Shuffle deterministically so the kept set is
    # a representative sample of the trained population, not an ordering artifact.
    order = np.random.default_rng(args.seed).permutation(len(eids))
    eids = [eids[i] for i in order]
    print(f"  found {len(eids)} candidate sessions; taking up to {args.max_sessions} that pass QC")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")

    used_sessions: list[dict] = []
    total_trials = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for eid in eids:
            if len(used_sessions) >= args.max_sessions:
                break
            try:
                trials = one.load_object(eid, "trials")
            except Exception as exc:  # noqa: BLE001 - a bad session should not abort the pull
                print(f"  skip {eid}: load failed ({exc})")
                continue
            trials = {k: np.asarray(v) if v is not None else None for k, v in dict(trials).items()}
            records, summary = session_to_records(trials, str(eid), args.rt_source)
            if len(records) < args.min_trials:
                print(f"  skip {eid}: only {len(records)} usable trials (< {args.min_trials})")
                continue
            easy_acc = summary["easy_full_contrast_accuracy"]
            if summary["n_full_contrast"] < args.min_full_contrast_trials:
                print(
                    f"  skip {eid}: only {summary['n_full_contrast']} full-contrast trials "
                    f"(< {args.min_full_contrast_trials}), cannot QC"
                )
                continue
            if easy_acc is None or easy_acc < args.min_easy_accuracy:
                print(
                    f"  skip {eid}: untrained — full-contrast accuracy {easy_acc:.3f} "
                    f"< {args.min_easy_accuracy}"
                )
                continue
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            total_trials += len(records)
            used_sessions.append(summary)
            flag = "" if summary["choice_sign_agreement"] >= 0.9 else "  ⚠ LOW choice-sign agreement"
            print(
                f"  + {eid}: {len(records)} trials, "
                f"full-contrast acc {summary['easy_full_contrast_accuracy']:.3f}, "
                f"choice-sign agreement {summary['choice_sign_agreement']:.3f}{flag}"
            )

    manifest = {
        "source": args.base_url,
        "task_protocol": args.task_protocol,
        "rt_source": args.rt_source,
        "contrast_set_abs": list(BIASED_BLOCK_ABS_CONTRASTS),
        "min_easy_accuracy": args.min_easy_accuracy,
        "selection_seed": args.seed,
        "n_sessions": len(used_sessions),
        "n_trials": total_trials,
        "sessions": used_sessions,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    low = [s for s in used_sessions if s["choice_sign_agreement"] < 0.9]
    print(f"\nWrote {total_trials} trials from {len(used_sessions)} sessions → {args.out}")
    print(f"Manifest → {manifest_path}")
    if low:
        print(f"⚠ {len(low)} session(s) had <0.90 choice-sign agreement — inspect before trusting.")
    print(
        "\nNext: compare targets without overwriting the baseline:\n"
        f"  python scripts/compute_reference_targets.py --reference-path {args.out}"
    )


def main() -> None:
    fetch(tyro.cli(Args))


if __name__ == "__main__":
    main()
