"""Freeze and execute a bounded generative history comparison on development animals."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_distribution
import tyro

from eval.generative_history import MODELS, fit_model, prepare, predict
from eval.metrics import load_trials


@dataclass(slots=True)
class Args:
    """New outputs only; no fresh cohort is acquired by this command."""

    output: Path = Path("runs/generative_history_20260919_v2")
    previous_plan: Path = Path("runs/competing_history_20260912/plan.json")
    repetitions: int = 40
    workers: int = 6
    maxiter: int = 1000
    resume: bool = False


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development(previous_plan: Path) -> tuple[pd.DataFrame, dict]:
    previous = json.loads(previous_plan.read_text())
    if digest(previous_plan) != previous_plan.with_suffix(".sha256").read_text().strip():
        raise ValueError("Previous plan hash mismatch")
    for path, expected in previous["input_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Changed historical input: {path}")
    identities = {s["id"]: s for s in json.loads(Path(previous["settings"]["identities"]).read_text())["sessions"]}
    raw = load_trials(Path(previous["settings"]["reference"]))
    for key in ("subject", "lab"):
        raw[key] = raw.session_id.map(lambda s: identities[s][key])
    if raw.subject.isin(previous["reserved_subjects_excluded"]).any():
        raise ValueError("Reserved animals overlap development inputs")
    return raw.sort_values(["session_id", "trial_index"]).reset_index(drop=True), previous


def simulate_template(raw: pd.DataFrame, regime: str, seed: int, strength: float = .15) -> tuple[pd.DataFrame, dict]:
    """Generate fresh responses on observed task schedules without using observed choices."""
    if regime not in ("static", "drift", "switching", "history", "mixed"):
        raise ValueError("Unknown regime")
    rng = np.random.default_rng(seed)
    data = raw.copy(deep=True)
    animal = {s: rng.normal(0,.4) for s in sorted(data.subject.unique())}
    lab = {s: rng.normal(0,.2) for s in sorted(data.lab.unique())}
    kl = []
    for _, group in data.groupby("session_id", sort=False):
        bias = animal[group.iloc[0].subject] + lab[group.iloc[0].lab] + rng.normal(0,.2)
        drift = rng.normal(0,.7)
        state = rng.integers(3)
        history: list[tuple[float,float,float]] = []
        previous = None
        for ordinal, i in enumerate(group.index):
            row = data.loc[i]
            if previous is not None and row.trial_index != previous+1:
                history = []
            previous = row.trial_index
            drift = .98*drift + rng.normal(0,.20)
            if rng.random() < .03:
                state = rng.integers(3)
            sensory = 4*np.sign(row.stimulus_contrast)*np.sqrt(abs(row.stimulus_contrast))
            block = np.log(row.block_prior['p_right']/(1-row.block_prior['p_right']))*.4
            z = sensory + block + bias
            if regime in ("drift", "mixed"):
                z += drift
            if regime == "switching":
                # Misspecified relative to bias-only drift: engagement also changes sensitivity.
                z = sensory*([1.,.35,.35][state]) + block + bias + [0.,-1.5,1.5][state]
            extra = 0.
            if history and regime in ("history", "mixed"):
                for lag, (a,c,w) in enumerate(reversed(history[-5:])):
                    extra += strength * (.6**lag) * a * (2*c-1)
                a,c,w = history[-1]
                extra += strength*a*w*(1-c)
            p, q = expit(z+extra), expit(z)
            if ordinal >= 100 and len(history) >= 5 and row.action in ("left","right"):
                kl.append(p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q)))
            if row.action not in ("left","right"):
                history = []
                continue
            action = 1. if rng.random() < p else -1.
            correct_side = np.sign(row.stimulus_contrast)
            if correct_side == 0:
                correct_side = 1. if rng.random() < row.block_prior['p_right'] else -1.
            correct = float(action == correct_side)
            data.at[i,"action"] = "right" if action == 1 else "left"
            data.at[i,"correct"] = bool(correct)
            history.append((action,correct,float(abs(row.stimulus_contrast)<=.125)))
    return data, {"regime":regime,"seed":seed,"strength":strength,
                  "oracle_conditional_history_kl":float(np.mean(kl)),
                  "note":"Oracle conditional KL is not the recoverable marginal model gain."}


def evaluate(raw: pd.DataFrame, folds: list[list[str]], maxiter: int, grid: int = 21) -> tuple[pd.DataFrame,list[dict]]:
    subjects = [s for fold in folds for s in fold]
    if len(subjects) != len(set(subjects)) or set(subjects) != set(raw.subject):
        raise ValueError("Animal-disjoint exhaustive folds required")
    results, fits = [], []
    for fold, heldout in enumerate(folds):
        train = prepare(raw[~raw.subject.isin(heldout)])
        test = prepare(raw[raw.subject.isin(heldout)])
        for model in MODELS:
            fit = fit_model(train, model, maxiter=maxiter, grid=grid)
            probability = np.clip(predict(test,fit),1e-12,1-1e-12)
            result = test.rows.loc[test.score,["subject","lab","session_id","trial_index"]].copy()
            y, p = test.y[test.score], probability[test.score]
            result["choice_right"], result["p_right"] = y, p
            result["nll"] = -y*np.log(p)-(1-y)*np.log1p(-p)
            result["model"],result["fold"] = model,fold
            results.append(result)
            fits.append({"fold":fold, "heldout":heldout, **asdict(fit)})
    return pd.concat(results,ignore_index=True),fits


def summarize(predictions: pd.DataFrame, seed: int = 20260919) -> dict:
    scores = predictions.groupby(["subject","lab","model"],as_index=False).nll.mean()
    wide = scores.pivot(index=["subject","lab"],columns="model",values="nll")
    gains = {}
    for base,full in [("task","history"),("drift","drift_history"),("switching","switching_history")]:
        d = (wide[base]-wide[full]).to_numpy()
        rng = np.random.default_rng(seed)
        # Resample labs, then animals within sampled labs; conditional on fitted models.
        labs = wide.index.get_level_values("lab").to_numpy()
        unique = np.unique(labs)
        boot = []
        for _ in range(2000):
            samples = []
            for lab in rng.choice(unique,len(unique),replace=True):
                values = d[labs==lab]
                samples.extend(rng.choice(values,len(values),replace=True))
            boot.append(np.mean(samples))
        gains[full] = {"mean":float(d.mean()),"interval":np.quantile(boot,[.025,.975]).tolist(),
                       "positive_animals":int((d>0).sum())}
    mean = scores.groupby("model").nll.mean().to_dict()
    return {"equal_animal_nll":mean,"winner":min(mean,key=mean.get),"gains":gains,
            "per_subject":scores.to_dict("records"),"animals":len(wide),
            "trials":len(predictions)//len(MODELS)}


def save(path: Path, value: object) -> None:
    def convert(obj: object) -> object:
        if isinstance(obj,np.ndarray):
            return obj.tolist()
        if isinstance(obj,np.generic):
            return obj.item()
        if isinstance(obj,Path):
            return str(obj)
        raise TypeError(type(obj))
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,default=convert)+"\n")
    temp.replace(path)


def recovery_job(job: tuple) -> dict:
    raw, regime, seed, strength, folds, maxiter, output = job
    path = Path(output)/f"recovery_{regime}_{strength}_{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    simulated, generator = simulate_template(raw,regime,seed,strength)
    predictions,fits = evaluate(simulated,folds,maxiter)
    summary = summarize(predictions,seed)
    summary.update(generator)
    summary["fits"] = fits
    summary["converged"] = all(f["converged"] for f in fits)
    # Conservative composite: history improves against BOTH latent alternatives.
    summary["history_detected"] = all(summary["gains"][m]["interval"][0]>0
                                      for m in ("drift_history","switching_history"))
    save(path,summary)
    return summary


def recovery_gate(recovery: list[dict]) -> dict:
    cells = {}
    expected_cells = {(r,0.) for r in ("static","drift","switching")}
    expected_cells |= {(r,s) for r in ("history","mixed") for s in (.05,.15,.4)}
    observed_cells = {(r["regime"],r["strength"]) for r in recovery}
    passed = observed_cells == expected_cells
    for regime,strength in sorted({(r["regime"],r["strength"]) for r in recovery}):
        group = [r for r in recovery if (r["regime"],r["strength"]) == (regime,strength)]
        hits = sum(r["history_detected"] for r in group)
        n = len(group)
        lo = 0. if hits == 0 else float(beta_distribution.ppf(.025,hits,n-hits+1))
        hi = 1. if hits == n else float(beta_distribution.ppf(.975,hits+1,n-hits))
        positive = regime in ("history","mixed")
        ok = (lo >= .8 if positive else hi <= .1) and all(r["converged"] for r in group)
        cells[f"{regime}_{strength}"] = {"runs":n,"detections":hits,"binomial_interval":[lo,hi],
                                         "pass":ok,"all_fits_converged":all(r["converged"] for r in group),
                                         "mean_oracle_kl":float(np.mean([r["oracle_conditional_history_kl"] for r in group]))}
        passed &= ok
    # Exact binomial intervals expose uncertainty in both specificity and power.
    return {"cells":cells,"recovery_pass":bool(passed),"fresh_cohort":"deferred",
            "reason":"All recovery cells, numerical stability, realistic power and external-method benchmarks must pass before a separate prospective freeze."}


def run(args: Args) -> None:
    if args.repetitions < 1 or args.workers < 1 or args.maxiter < 1:
        raise ValueError("Positive repetitions, workers and iteration budget required")
    raw, previous = load_development(args.previous_plan)
    sources = [Path(__file__),Path("eval/generative_history.py"),Path("eval/bias_history.py"),
               Path("eval/history_audit.py"),Path("eval/metrics.py"),Path("docs/GENERATIVE_COMPARISON_PROTOCOL.md")]
    source_hashes = {str(p.resolve().relative_to(Path.cwd())):digest(p) for p in sources}
    # One fixed split for repeated recovery; full five-fold real development evaluation.
    shuffled = np.random.default_rng(20260919).permutation(sorted(raw.subject.unique()))
    recovery_folds = [x.tolist() for x in np.array_split(shuffled,2)]
    settings = asdict(args)
    settings.pop("resume")
    settings.pop("workers")
    plan = {"settings":settings,"source_sha256":source_hashes,"input_sha256":previous["input_sha256"],
            "previous_plan_sha256":digest(args.previous_plan),"folds":previous["subject_folds"],
            "recovery_folds":recovery_folds,"reserved_subjects_excluded":previous["reserved_subjects_excluded"],
            "models":MODELS,"prefix":100,"min_prefix_choices":80,"common_history":5,
            "drift":{"grid":21,"range":[-3,3],"rho":.98,"innovation_sd":.20},
            "switching":{"states":3,"starts":[.5,1.5],"initialization":"training GLM plus state intercept offsets","train_only_transition_fit":True},
            "recovery":{"regimes":["static","drift","switching","history","mixed"],
                        "history_strengths":[.05,.15,.4],"seeds":list(range(1900,1900+args.repetitions)),
                        "template":"All development animals, sessions, task schedules and omission positions; new responses and lab/animal/session heterogeneity"},
            "decision":{"null_upper_95_max":.1,"power_lower_95_min":.8,
                        "all_optimizers_must_converge":True,"grid_sensitivity_required":True,
                        "published_implementation_benchmarks_required":True},
            "status":"exploratory development; internal pre-run freeze, not external preregistration",
            "runtime":{"python":platform.python_version(),"numpy":np.__version__}}
    args.output.mkdir(parents=True,exist_ok=True)
    plan_path = args.output/"plan.json"
    if plan_path.exists():
        if not args.resume:
            raise ValueError("Existing run requires explicit resume")
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan,default=str)):
            raise ValueError("Cannot resume changed settings, source, or data")
        if digest(plan_path) != plan_path.with_suffix('.sha256').read_text().strip():
            raise ValueError("Plan changed")
    elif any(args.output.iterdir()):
        raise ValueError("Output directory is not empty")
    else:
        save(plan_path,plan)
        plan_path.with_suffix('.sha256').write_text(digest(plan_path)+'\n')
    print('Plan frozen; running development comparison',flush=True)
    if not (args.output/'development.json').exists():
        predictions,fits = evaluate(raw,previous['subject_folds'],args.maxiter)
        predictions.to_csv(args.output/'predictions.csv',index=False)
        save(args.output/'development.json',{'summary':summarize(predictions),'fits':fits})
    jobs = []
    for regime in plan['recovery']['regimes']:
        for strength in ([.05,.15,.4] if regime in ('history','mixed') else [0.]):
            for seed in plan['recovery']['seeds']:
                jobs.append((raw,regime,seed,strength,recovery_folds,args.maxiter,str(args.output)))
    recovery = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(recovery_job,jobs):
            recovery.append(result)
            print(f"Recovery {len(recovery)}/{len(jobs)}: {result['regime']} {result['strength']} seed={result['seed']}",flush=True)
    gate = recovery_gate(recovery)
    save(args.output/'decision.json',gate)
    for path,expected in {**source_hashes,**previous['input_sha256']}.items():
        if digest(Path(path)) != expected:
            raise ValueError('Source or input changed during run')
    save(args.output/'complete.json',{'plan_sha256':digest(plan_path),'decision':gate})


if __name__ == '__main__':
    run(tyro.cli(Args))
