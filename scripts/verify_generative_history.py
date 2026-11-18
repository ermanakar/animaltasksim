"""Independently reconstruct saved causal predictions and model-comparison summaries."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
import tyro

from scripts.compare_generative_history import digest, load_development, save, summarize
from eval.history_audit import CONTRASTS, PRIORS


@dataclass(slots=True)
class Args:
    output: Path = Path("runs/generative_history_20260919_v2")


def verify(args: Args) -> None:
    plan = json.loads((args.output/'plan.json').read_text())
    if digest(args.output/'plan.json') != (args.output/'plan.sha256').read_text().strip():
        raise ValueError('Plan hash mismatch')
    for path,expected in {**plan['source_sha256'],**plan['input_sha256']}.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Changed input {path}')
    raw,_ = load_development(Path(plan['settings']['previous_plan']))
    fits = json.loads((args.output/'development.json').read_text())['fits']
    predictions = pd.read_csv(args.output/'predictions.csv')
    grouped = raw.groupby('session_id',sort=False)
    ordinal = grouped.cumcount()
    valid = raw.action.isin(['left','right'])
    counts = valid[ordinal<100].groupby(raw.loc[ordinal<100,'session_id']).sum()
    eligible = valid & (ordinal>=100) & (raw.session_id.map(counts)>=80)
    histories = []
    for lag in range(1,6):
        actions = grouped.action.shift(lag)
        successes = grouped.correct.shift(lag)
        eligible &= actions.isin(['left','right']) & (raw.trial_index-grouped.trial_index.shift(lag)==lag)
        signed = np.where(actions=='right',1.,-1.)
        # A lag feature is available only if every intervening predecessor is committed/contiguous.
        available = valid.copy()
        for j in range(1,lag+1):
            available &= grouped.action.shift(j).isin(['left','right']) & (raw.trial_index-grouped.trial_index.shift(j)==j)
        histories.extend([np.where(available,signed*(successes==True),0.),  # noqa: E712
                          np.where(available,signed*(successes==False),0.)])  # noqa: E712
    previous_weak = (grouped.stimulus_contrast.shift(1).abs()<=.125).to_numpy()
    histories.extend([histories[0]*previous_weak,histories[1]*previous_weak])
    h = np.column_stack(histories)
    task = [np.ones(len(raw))]
    for contrast in CONTRASTS:
        for prior in PRIORS:
            if contrast == 0 and prior == .5:
                continue
            task.append(np.isclose(raw.stimulus_contrast,contrast,atol=1e-8,rtol=0) &
                        np.isclose(raw.block_prior.map(lambda b:b['p_right']),prior,atol=1e-8,rtol=0))
    x = np.column_stack(task)
    max_error = 0.
    covered = 0
    for fit in fits:
        model,fold = fit['model'],fit['fold']
        beta,a,offsets = map(np.asarray,(fit['beta'],fit['transition'],fit['offsets']))
        design = np.column_stack([x,h]) if 'history' in model else x
        observations = predictions[(predictions.model==model)&(predictions.fold==fold)].set_index(['session_id','trial_index'])
        expected_keys = pd.MultiIndex.from_frame(raw.loc[eligible & raw.subject.isin(fit['heldout']),['session_id','trial_index']])
        if set(observations.index) != set(expected_keys):
            raise ValueError('Model scoring rows differ from independently reconstructed common sample')
        for sid,group in raw[raw.subject.isin(fit['heldout'])].groupby('session_id',sort=False):
            belief = np.ones(len(offsets))/len(offsets)
            previous = None
            for i,row in group.iterrows():
                if previous is None or row.trial_index != previous+1:
                    belief = np.ones(len(offsets))/len(offsets)
                previous = row.trial_index
                if row.action not in ('left','right'):
                    belief = np.ones(len(offsets))/len(offsets)
                    continue
                p_state = np.clip(expit(design[i]@beta.T+offsets),1e-12,1-1e-12)
                p = float(belief@p_state)
                if eligible[i]:
                    saved = observations.loc[(sid,row.trial_index)]
                    max_error = max(max_error,abs(saved.p_right-p))
                    if saved.choice_right != (row.action=='right'):
                        raise ValueError('Saved response differs from source')
                    covered += 1
                likelihood = p_state if row.action=='right' else 1-p_state
                posterior = belief*likelihood
                posterior /= posterior.sum()
                belief = posterior@a
    if covered != len(predictions) or max_error>1e-8:
        raise ValueError(f'Prediction reconstruction failed: {covered=} {max_error=}')
    y,p = predictions.choice_right.to_numpy(),predictions.p_right.to_numpy()
    loss = -y*np.log(p)-(1-y)*np.log1p(-p)
    np.testing.assert_allclose(predictions.nll,loss,atol=1e-12,rtol=1e-10)
    saved = json.loads((args.output/'development.json').read_text())['summary']
    recalculated = summarize(predictions)
    for model,value in recalculated['equal_animal_nll'].items():
        if abs(saved['equal_animal_nll'][model]-value)>1e-12:
            raise ValueError('Summary mismatch')
    save(args.output/'verification.json',{'prediction_rows':covered,'max_probability_error':max_error,
          'source_and_plan_hashes':'passed','common_scoring_rows':'passed','saved_losses_and_summary':'passed',
          'verifier_sha256':digest(Path(__file__))})


def verify_recovery(output: Path) -> None:
    """Check complete cell coverage, saved means, selection flags, and fit diagnostics."""
    plan = json.loads((output/'plan.json').read_text())
    expected = {(r,s,seed) for r in plan['recovery']['regimes']
                for s in (plan['recovery']['history_strengths'] if r in ('history','mixed') else [0.])
                for seed in plan['recovery']['seeds']}
    seen = set()
    largest = 0.
    failed_starts = 0
    subjects = {s for fold in plan['recovery_folds'] for s in fold}
    expected_trials = json.loads((output/'development.json').read_text())['summary']['trials']
    for path in output.glob('recovery_*.json'):
        if path.name == 'recovery_verification.json':
            continue
        result = json.loads(path.read_text())
        key = (result['regime'],result['strength'],result['seed'])
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unexpected recovery cell')
        seen.add(key)
        scores = pd.DataFrame(result['per_subject'])
        if scores.duplicated(['subject','lab','model']).any() or set(scores.subject)!=subjects:
            raise ValueError('Invalid recovery scoring units')
        if result['trials'] != expected_trials or len(scores)!=len(subjects)*len(plan['models']):
            raise ValueError('Recovery cohort or scoring sample changed')
        means = scores.groupby('model').nll.mean().to_dict()
        for model,value in means.items():
            largest = max(largest,abs(value-result['equal_animal_nll'][model]))
        if result['winner'] != min(means,key=means.get):
            raise ValueError('Incorrect model selection label')
        detected = all(result['gains'][m]['interval'][0]>0 for m in ('drift_history','switching_history'))
        if result['history_detected'] != detected:
            raise ValueError('Incorrect history detection label')
        if result['converged'] != all(f['converged'] for f in result['fits']):
            raise ValueError('Incorrect convergence label')
        pairs = {(f['fold'],f['model']) for f in result['fits']}
        if pairs != {(i,m) for i in range(len(plan['recovery_folds'])) for m in plan['models']}:
            raise ValueError('Missing model fits')
        for fit in result['fits']:
            if fit['heldout'] != plan['recovery_folds'][fit['fold']]:
                raise ValueError('Changed recovery folds')
            if abs(fit['objective']-min(a['objective'] for a in fit['starts']))>1e-8:
                raise ValueError('Start selection not based on training objective')
            failed_starts += sum(not a['success'] for a in fit['starts'])
            if not np.isfinite(np.asarray(fit['beta'])).all():
                raise ValueError('Nonfinite fitted coefficients')
            np.testing.assert_allclose(np.asarray(fit['transition']).sum(1),1.,atol=1e-12)
    if seen != expected or largest>1e-12:
        raise ValueError('Incomplete recovery or inconsistent summary')
    save(output/'recovery_verification.json',{'datasets':len(seen),'max_mean_nll_error':largest,
          'complete_cell_coverage':'passed','subject_and_scoring_counts':'passed',
          'model_selection_and_convergence_labels':'passed','failed_optimizer_starts':failed_starts,
          'limits':'Checks saved fits and summaries; does not independently regenerate all synthetic responses or refit models.'})


if __name__ == '__main__':
    args = tyro.cli(Args)
    verify(args)
    if (args.output/'complete.json').exists():
        verify_recovery(args.output)
