def _py(v):
    """Convert NumPy scalar types to native Python types so json.dumps won't crash."""
    try:
        import numpy as _np
        if isinstance(v, (_np.floating, _np.integer)):
            return v.item()
        if isinstance(v, _np.bool_):
            return bool(v)
    except Exception:
        pass
    return v

def _clean_dict(d: dict) -> dict:
    return {k: _py(v) if not isinstance(v, dict) else _clean_dict(v) for k, v in d.items()}
#!/usr/bin/env python
"""
Fetch one IBL mouse session and export to data/ibl/reference_single_session.ndjson
"""
import json
import math
import argparse
from pathlib import Path
from one.api import ONE
import requests, io
import numpy as np

# Helper for safe NaN checks
def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)

FLATIRON = "https://ibl.flatironinstitute.org/public/"

def http_load_trials_from_flatiron(eid: str) -> dict:
    """
    Minimal public HTTP loader for a subset of trials fields, avoiding Alyx auth.
    Downloads a few `_ibl_trials.*.npy` arrays directly from FlatIron.
    Missing arrays are set to None.
    """
    rel = [
        "alf/_ibl_trials.contrastLeft.npy",
        "alf/_ibl_trials.contrastRight.npy",
        "alf/_ibl_trials.choice.npy",
        "alf/_ibl_trials.feedbackType.npy",
        "alf/_ibl_trials.goCue_times.npy",
        "alf/_ibl_trials.response_times.npy",
        "alf/_ibl_trials.probabilityLeft.npy",
    ]
    base = f"{FLATIRON}alyx/{eid}/"
    out: dict[str, object] = {}
    for r in rel:
        url = base + r
        key = r.split("/")[-1].replace("_ibl_trials.", "").replace(".npy", "")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200 or not resp.content:
                out[key] = None
                continue
            arr = np.load(io.BytesIO(resp.content))
            out[key] = arr
        except Exception:
            out[key] = None
    return out

def signed_contrast(trials, i):
    cl = trials.get('contrastLeft', [None])[i]
    cr = trials.get('contrastRight', [None])[i]
    # Keep 0.0 values; treat only real NaNs as missing
    if cr is not None and not _is_nan(cr):
        return _py(float(cr))
    if cl is not None and not _is_nan(cl):
        return _py(-float(cl))
    return None

def p_right(trials, i):
    pL = trials.get('probabilityLeft', [None])[i]
    if pL is None or _is_nan(pL):
        return None
    return _py(float(1.0 - pL))

def map_choice(val):
    # IBL choices: -1=left, +1=right, 0=no-go (rare)
    if val in (-1, -1.0):
        code = 0  # left
    elif val in (1, 1.0):
        code = 1  # right
    else:
        code = 2      # no-op
    return _py(code)

def _trials_len(trials: dict) -> int:
    candidates = []
    for k in ("choice", "contrastLeft", "contrastRight", "feedbackType"):
        arr = trials.get(k)
        if arr is not None:
            try:
                candidates.append(len(arr))
            except Exception:
                pass
    return max(candidates) if candidates else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", default=None, help="IBL session EID (UUID); bypasses search/auth")
    ap.add_argument("--subject", default=None, help="Subject nickname (e.g., KS022)")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--session", type=int, default=1, help="Session number on that date (1-based)")
    ap.add_argument("--out", type=Path, default=Path("data/ibl/reference_single_session.ndjson"))
    args = ap.parse_args()

    # Public access: documented in ONE quick-start (intbrainlab / international)
    one = ONE(base_url='https://openalyx.internationalbrainlab.org',
              silent=True,
              username='intbrainlab',
              password='international')
    if args.eid:
        eid = args.eid
    else:
        if not (args.subject and args.date):
            raise SystemExit("Provide --eid OR both --subject and --date")
        eids = one.search(subject=args.subject, date_range=args.date)
        if not eids:
            raise SystemExit(f"No sessions found for subject={args.subject} date={args.date}")
        try:
            eid = eids[args.session - 1]
        except IndexError:
            raise SystemExit(f"Session {args.session} not found; available count: {len(eids)}")

    try:
        trials = one.load_object(eid, 'trials')
    except Exception:
        print("⚠️ Alyx auth blocked or API error; falling back to public HTTP (FlatIron) for trials …")
        trials = http_load_trials_from_flatiron(eid)
    session_id = str(eid)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        prev = None
        for i in range(_trials_len(trials)):
            # Safety: skip trials if choice or feedbackType missing
            if trials.get("choice") is None or trials.get("feedbackType") is None:
                continue
            act = _py(map_choice(trials["choice"][i]))
            corr = _py(bool(trials["feedbackType"][i] == 1))
            rt_val = (
                float((trials["response_times"][i] - trials["goCue_times"][i]) * 1000.0)
                if (trials.get("goCue_times") is not None
                    and trials.get("response_times") is not None
                    and not _is_nan(trials["goCue_times"][i])
                    and not _is_nan(trials["response_times"][i]))
                else None
            )
            record = {
                "task": "IBL2AFC",
                "session_id": session_id,
                "trial_index": int(i),
                "stimulus": {"contrast": signed_contrast(trials, i)},
                "block_prior": {"p_right": p_right(trials, i)},
                "action": act,
                "correct": corr,
                "reward": _py(1.0 if corr else 0.0),
                "rt_ms": _py(rt_val),
                "phase_times": {},  # required by TrialRecord schema; values not used for IBL reference
                "prev": prev,
                "seed": 0,
                "agent": {"name": "reference_mouse", "version": "ibl_public"}
            }
            record = _clean_dict(record)
            f.write(json.dumps(record, separators=(",", ":")) + "\n"); f.flush()
            prev = {"action": record["action"], "reward": record["reward"], "correct": record["correct"]}
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    main()
