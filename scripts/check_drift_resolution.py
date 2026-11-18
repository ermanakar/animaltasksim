"""Supplementary fixed grid-resolution and solver sensitivity on development animals."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro

from eval.generative_history import fit_model, prepare, predict
from scripts.compare_generative_history import digest, load_development, save


@dataclass(slots=True)
class Args:
    output: Path = Path('runs/generative_history_20260919_v2')


def run(args: Args) -> None:
    raw, previous = load_development(Path('runs/competing_history_20260912/plan.json'))
    plan_path = args.output/'resolution_plan.json'
    plan = {'grids':[21,41],'models':['drift','drift_history'],'maxiter':1000,
            'folds':previous['subject_folds'],'threshold_max_abs_equal_animal_nll_shift':.0001,
            'source_sha256':{str(Path(__file__).relative_to(Path.cwd())):digest(Path(__file__)),
                             'eval/generative_history.py':digest(Path('eval/generative_history.py'))},
            'scope':'Supplementary fixed numerical sensitivity; does not revise primary models or select a grid using test performance.'}
    if plan_path.exists():
        raise ValueError('Do not overwrite sensitivity result')
    save(plan_path,plan)
    rows = []
    for fold,subjects in enumerate(previous['subject_folds']):
        train = prepare(raw[~raw.subject.isin(subjects)])
        test = prepare(raw[raw.subject.isin(subjects)])
        for grid in plan['grids']:
            for model in plan['models']:
                fit = fit_model(train,model,maxiter=1000,grid=grid)
                p = np.clip(predict(test,fit),1e-12,1-1e-12)
                loss = -test.y*np.log(p)-(1-test.y)*np.log1p(-p)
                scores = test.rows.loc[test.score,['subject']].copy()
                scores['nll'] = loss[test.score]
                rows.append({'grid':grid,'model':model,'fold':fold,'fit':asdict(fit),
                             'per_subject':scores.groupby('subject').nll.mean().to_dict()})
    changes = {}
    for model in plan['models']:
        means = {grid:float(np.mean([v for r in rows if r['grid']==grid and r['model']==model
                                     for v in r['per_subject'].values()])) for grid in plan['grids']}
        changes[model] = {'equal_animal_nll':means,'difference_41_minus_21':means[41]-means[21]}
    save(args.output/'resolution_check.json',{'results':rows,'changes':changes,
         'pass':all(abs(v['difference_41_minus_21'])<=.0001 for v in changes.values()) and all(r['fit']['converged'] for r in rows),
         'plan_sha256':digest(plan_path)})


if __name__ == '__main__':
    run(tyro.cli(Args))
