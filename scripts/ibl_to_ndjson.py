#!/usr/bin/env python
"""
Fetch one IBL mouse session and export to data/ibl/reference.ndjson
"""
import json
import math
import argparse
from pathlib import Path
from one.api import ONE

def signed_contrast(trials, i):
    cl, cr = trials.get('contrastLeft',[None])[i], trials.get('contrastRight',[None])[i]
    if cr and not math.isnan(cr): return cr
    if cl and not math.isnan(cl): return -cl
    return None

def p_right(trials, i):
    pL = trials.get('probabilityLeft',[None])[i]
    if pL is None or (isinstance(pL,float) and math.isnan(pL)): return None
    return 1 - pL

def map_choice(val):
    if val == -1: return 0  # left
    if val == 1: return 1   # right
    return 2                # no-go

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="KS022")
    ap.add_argument("--date", default="2019-12-10")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--out", default="data/ibl/reference.ndjson")
    args = ap.parse_args()

    one = ONE(base_url='https://openalyx.internationalbrainlab.org')
    eids = one.search(subject=args.subject, date_range=args.date)
    eid = eids[args.session - 1]

    trials = one.load_object(eid, 'trials')
    session_id = str(eid)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        prev = None
        for i in range(len(trials['choice'])):
            record = {
                "task": "IBL2AFC",
                "session_id": session_id,
                "trial_index": i,
                "stimulus": {"contrast": signed_contrast(trials,i)},
                "block_prior": {"p_right": p_right(trials,i)},
                "action": map_choice(trials["choice"][i]),
                "correct": trials["feedbackType"][i] == 1,
                "reward": 1.0 if trials["feedbackType"][i] == 1 else 0.0,
                "rt_ms": (trials["response_times"][i]-trials["goCue_times"][i])*1000 if not math.isnan(trials["goCue_times"][i]) else None,
                "prev": prev,
                "seed": 0,
                "agent": {"name": "reference_mouse", "version": "ibl_public"}
            }
            f.write(json.dumps(record)+"\n")
            prev = {"action": record["action"], "reward": record["reward"], "correct": record["correct"]}
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    main()