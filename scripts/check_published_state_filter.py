"""Cross-check causal predictions using the GLM-HMM authors' pinned ssm package."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import expit
import tyro

from eval.generative_history import Fit, prepare, predict
from scripts.compare_generative_history import digest, load_development, save


@dataclass(slots=True)
class Args:
    output: Path = Path('runs/generative_history_20260919_v2')
    vendor: Path = Path('runs/vendor_ssm_20260919')


def run(args: Args) -> None:
    from ssm.messages import hmm_filter
    from ssm.observations import InputDrivenObservations
    import ssm
    expected = 'e9b408b79e2ab22a05ca93c3a1f78a7dae461992'
    revision = subprocess.check_output(['git','-C',str(args.vendor),'rev-parse','HEAD'],text=True).strip()
    if revision != expected:
        raise ValueError('Unreviewed upstream revision')
    installed = Path(ssm.__file__).parent
    for name in ('messages.py','observations.py'):
        if digest(installed/name) != digest(args.vendor/'ssm'/name):
            raise ValueError('Installed upstream code differs from pin')
    raw,_ = load_development(Path('runs/competing_history_20260912/plan.json'))
    results = json.loads((args.output/'development.json').read_text())
    errors,rows = [],0
    for fitted in results['fits']:
        if not fitted['model'].startswith('switching'):
            continue
        seq = prepare(raw[raw.subject.isin(fitted['heldout'])])
        fit = Fit(**{k:np.asarray(v) if k in ('beta','transition','offsets') else v
                     for k,v in fitted.items() if k not in ('fold','heldout')})
        x = np.column_stack([seq.x,seq.h]) if 'history' in fit.model else seq.x
        observation = InputDrivenObservations(3,1,M=x.shape[1],C=2,prior_sigma=1)
        # Author code assigns learned logits to class zero, so invert our right=1 weights.
        observation.params = -fit.beta[:,None,:]
        logits = observation.calculate_logits(x)
        np.testing.assert_allclose(np.exp(logits[:,:,1]),expit(x@fit.beta.T),atol=1e-12)
        ll = observation.log_likelihoods(seq.y.astype(int)[:,None],x,None,None)
        probability = np.empty(len(seq.y))
        starts = np.flatnonzero(seq.reset)
        for start,end in zip(starts,np.r_[starts[1:],len(seq.y)]):
            prior = hmm_filter(np.ones(3)/3,fit.transition[None,:,:],np.ascontiguousarray(ll[start:end]))
            probability[start:end] = (prior*np.exp(logits[start:end,:,1])).sum(1)
        error = float(np.max(np.abs(probability-predict(seq,fit))))
        errors.append({'model':fit.model,'fold':fitted['fold'],'max_probability_error':error})
        rows += len(seq.y)
    if max(r['max_probability_error'] for r in errors)>1e-9:
        raise ValueError('Published implementation disagrees')
    save(args.output/'published_filter_check.json',{'upstream':'https://github.com/zashwood/ssm',
          'revision':revision,'rows':rows,'errors':errors,'script_sha256':digest(Path(__file__)),
          'scope':'Independent author-code emissions and causal filter at identical saved parameters. Does not independently refit models or reproduce the paper.'})


if __name__ == '__main__':
    run(tyro.cli(Args))
