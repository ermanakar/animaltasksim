"""Independently refit matched GLM-HMMs using the authors' likelihood derivatives."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
import tyro

from eval.generative_history import Fit, _objective, fit_model, prepare, predict
from scripts.compare_generative_history import digest, load_development, save


@dataclass(slots=True)
class Args:
    output: Path = Path('runs/generative_history_20260919_v2')


def author_objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray, reset: np.ndarray) -> float:
    """Use upstream's autograd HMM normalizer, with the primary model's priors."""
    import autograd.numpy as anp
    from autograd.scipy.special import logsumexp
    from ssm.primitives import hmm_normalizer
    p = x.shape[1]
    weights = anp.reshape(theta[:3*p],(3,p))
    logits = anp.reshape(theta[3*p:],(3,3))
    log_a = logits-logsumexp(logits,axis=1,keepdims=True)
    a = anp.exp(log_a)
    # Uniform transitions at reset boundaries remove all dependence on preceding state.
    transitions = anp.where(reset[1:,None,None],anp.ones((len(y)-1,3,3))/3,
                            anp.tile(a,(len(y)-1,1,1)))
    z = anp.dot(x,weights.T)
    ll = anp.where(y[:,None]==1,-anp.logaddexp(0,-z),-anp.logaddexp(0,z))
    return -hmm_normalizer(anp.ones(3)/3,transitions,ll)+.5*anp.sum(weights*weights)-.1*anp.sum(log_a)


def run(args: Args) -> None:
    from autograd import value_and_grad
    import ssm
    reference = json.loads((args.output/'published_filter_check.json').read_text())
    installed = Path(ssm.__file__).parent
    rng = np.random.default_rng(19)
    x,y = rng.normal(size=(30,4)),rng.integers(2,size=30).astype(float)
    reset = np.zeros(30,dtype=bool)
    reset[[0,10,20]] = True
    theta = rng.normal(size=21)*.2
    value,grad = value_and_grad(author_objective)(theta,x,y,reset)
    ours,our_grad = _objective(theta,x,y,reset,'switching',np.zeros(3),np.eye(3))
    np.testing.assert_allclose(value,ours,atol=1e-8)
    np.testing.assert_allclose(grad,our_grad,atol=1e-7)
    raw,previous = load_development(Path('runs/competing_history_20260912/plan.json'))
    plan = {'upstream':reference['upstream'],'revision':reference['revision'],
            'likelihood_source_sha256':digest(installed/'primitives.py'),
            'script_sha256':digest(Path(__file__)), 'models':['switching','switching_history'],
            'folds':previous['subject_folds'],'maxiter':1000,'starts':[.5,1.5],
            'scope':'Matched features, reset rules and priors; independently evaluated upstream likelihood and gradients, same scipy optimizer. Not a reproduction of paper preprocessing or cohort.',
            'qualification_max_abs_equal_animal_nll_difference':.0001}
    path = args.output/'author_refit_plan.json'
    if path.exists():
        raise ValueError('Do not overwrite independent refit')
    save(path,plan)
    fits,records = [],[]
    for fold,subjects in enumerate(plan['folds']):
        train = prepare(raw[~raw.subject.isin(subjects)])
        test = prepare(raw[raw.subject.isin(subjects)])
        for model in plan['models']:
            history = model.endswith('history')
            train_x = np.column_stack([train.x,train.h]) if history else train.x
            base = fit_model(train,'history' if history else 'task',maxiter=1000).beta[0]
            attempts = []
            for strength in plan['starts']:
                weights = np.tile(base,(3,1))
                weights[:,0] += np.linspace(-strength,strength,3)
                theta = np.r_[weights.ravel(),np.log(.97*np.eye(3)+.03/3).ravel()]
                result = minimize(value_and_grad(author_objective),theta,
                                  args=(train_x,train.y,train.reset),method='L-BFGS-B',jac=True,
                                  options={'maxiter':1000,'ftol':1e-9,'gtol':1e-4,'maxls':30})
                attempts.append(result)
            result = min(attempts,key=lambda r:r.fun)
            p = train_x.shape[1]
            fit = Fit(model,result.x[:3*p].reshape(3,p),softmax(result.x[3*p:].reshape(3,3),axis=1),
                      np.zeros(3),float(result.fun),bool(result.success),int(result.nit),
                      float(np.max(np.abs(result.jac))),[])
            probability = np.clip(predict(test,fit),1e-12,1-1e-12)
            loss = -test.y*np.log(probability)-(1-test.y)*np.log1p(-probability)
            scores = test.rows.loc[test.score,['subject','lab']].copy()
            scores['nll'],scores['model'] = loss[test.score],model
            records.append(scores)
            fits.append({'fold':fold,'model':model,'beta':fit.beta,'transition':fit.transition,
                         'objective':fit.objective,'converged':fit.converged,'iterations':fit.iterations,
                         'starts':[{'objective':float(r.fun),'converged':bool(r.success),'iterations':int(r.nit)} for r in attempts]})
            print('Author refit',fold,model,fit.converged,flush=True)
    scores = pd.concat(records).groupby(['subject','lab','model'],as_index=False).nll.mean()
    means = scores.groupby('model').nll.mean().to_dict()
    original = json.loads((args.output/'development.json').read_text())['summary']['equal_animal_nll']
    differences = {m:v-original[m] for m,v in means.items()}
    save(args.output/'author_refit.json',{'plan':plan,'fits':fits,'equal_animal_nll':means,
          'per_subject':scores.to_dict('records'),'difference_from_primary':differences,
          'pass':all(abs(d)<=.0001 for d in differences.values()) and all(f['converged'] for f in fits)})


if __name__ == '__main__':
    run(tyro.cli(Args))
