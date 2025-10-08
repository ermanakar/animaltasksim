#!/usr/bin/env python3
"""Convert Roitman & Shadlen (2002) macaque RDM CSV to AnimalTaskSim .ndjson schema."""
import argparse
import csv
import json
import pathlib


def signed_coh(row):
    """
    Convert coherence magnitude + target choice to signed coherence.
    'coh' is magnitude; 'trgchoice' tells which side was correct.
    Map trgchoice==2 -> positive (right), trgchoice==1 -> negative (left).
    """
    c = float(row["coh"])
    tc = float(row.get("trgchoice", 2.0))
    return c if tc == 2.0 else -c


def infer_action(s_coh, correct_flag):
    """
    Infer the action string from signed coherence and correctness.
    If correct==1, action matches sign(coh). Else, it's the opposite.
    Returns "left" or "right" string to match schema.
    """
    if correct_flag:
        return "right" if s_coh > 0 else "left"
    else:
        return "left" if s_coh > 0 else "right"


def main():
    ap = argparse.ArgumentParser(
        description="Convert Roitman & Shadlen macaque RDM CSV to .ndjson"
    )
    ap.add_argument("--csv", required=True, help="Path to roitman_rts.csv")
    ap.add_argument(
        "--monkey",
        type=int,
        choices=[1, 2],
        default=1,
        help="Which monkey (1 or 2) to export (default: 1)",
    )
    ap.add_argument("--out", required=True, help="Output .ndjson file path")
    args = ap.parse_args()

    n = 0
    prev = None
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(args.csv, "r") as f, open(outp, "w") as g:
        r = csv.DictReader(f)
        for row in r:
            if int(float(row["monkey"])) != args.monkey:
                continue
            # Basic sanity filters from PyDDM quickstart: keep 0.1s < RT < 1.65s
            rt_s = float(row["rt"])
            if not (0.1 <= rt_s <= 1.65):
                continue

            s_coh = signed_coh(row)
            corr = bool(int(float(row["correct"])))
            action = infer_action(s_coh, corr)

            rec = {
                "task": "rdm",  # lowercase for consistency
                "session_id": f"roitman_shadlen_2002_M{args.monkey}",
                "trial_index": n,
                "stimulus": {"coherence": float(s_coh), "direction": "right" if s_coh > 0 else "left"},
                "block_prior": None,  # RDM task has no block structure
                "action": action,  # "left" or "right" string
                "correct": bool(corr),
                "reward": 1.0 if corr else 0.0,
                "rt_ms": float(rt_s * 1000.0),
                "phase_times": {},  # required by schema but not available from CSV
                "prev": prev,
                "seed": 0,  # unknown from CSV
                "agent": {
                    "name": "macaque_reference",
                    "version": "roitman_shadlen_2002",
                },
            }
            g.write(json.dumps(rec) + "\n")
            prev = {
                "action": rec["action"],
                "reward": rec["reward"],
                "correct": rec["correct"],
            }
            n += 1

    print(f"✅ Wrote {n} trials to {args.out}")


if __name__ == "__main__":
    main()
