"""Publish compact results and a figure without refitting or selecting experiments."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

from scripts.compare_generative_history import digest, save


@dataclass(slots=True)
class Args:
    output: Path = Path('runs/generative_history_20260919_v2')
    report: Path = Path('docs/GENERATIVE_COMPARISON.md')


def run(args: Args) -> None:
    plan = json.loads((args.output/'plan.json').read_text())
    completed = json.loads((args.output/'complete.json').read_text())
    if completed['plan_sha256'] != digest(args.output/'plan.json'):
        raise ValueError('Plan hash mismatch')
    development = json.loads((args.output/'development.json').read_text())
    decision = json.loads((args.output/'decision.json').read_text())
    recovery = [json.loads(p.read_text()) for p in sorted(args.output.glob('recovery_*_*.json'))]
    summaries = [{k:v for k,v in r.items() if k not in ('fits','per_subject')} for r in recovery]
    winners = {cell:dict(Counter(r['winner'] for r in recovery if f"{r['regime']}_{r['strength']}"==cell))
               for cell in decision['cells']}
    checks = {name:json.loads((args.output/name).read_text()) if (args.output/name).exists() else {'status':'not completed'}
              for name in ('verification.json','published_filter_check.json','resolution_check.json','author_refit.json')}
    failed_fits = sum(not f['converged'] for r in recovery for f in r['fits'])
    real_failed = sum(not f['converged'] for f in development['fits'])
    failed_starts = sum(not a['success'] for r in recovery for f in r['fits'] for a in f['starts'])
    real_failed_starts = sum(not a['success'] for f in development['fits'] for a in f['starts'])
    bundle = {'plan':plan,'plan_sha256':digest(args.output/'plan.json'),
              'artifact_sha256':{p.name:digest(p) for p in sorted(args.output.glob('*.json'))},
              'environment':json.loads((args.output/'environment.json').read_text()),
              'recovery_verification':json.loads((args.output/'recovery_verification.json').read_text()),
              'development':development['summary'],'development_fits':development['fits'],
              'recovery':summaries,'decision':decision,
              'winner_frequencies':winners,'recovery_failed_fits':failed_fits,
              'development_failed_fits':real_failed,'recovery_failed_optimizer_starts':failed_starts,
              'development_failed_optimizer_starts':real_failed_starts,'verification':checks['verification.json'],
              'published_filter_check':checks['published_filter_check.json'],
              'author_refit_summary':{k:v for k,v in checks['author_refit.json'].items() if k!='fits'},
              'resolution_summary':{k:v for k,v in checks['resolution_check.json'].items() if k!='results'}}
    save(Path('docs/results/generative_comparison_v1.json'),bundle)
    means = development['summary']['equal_animal_nll']
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    models = sorted(means,key=means.get,reverse=True)
    axes[0].barh(models,[means[m] for m in models],color='#386b8d')
    axes[0].set_xlabel('Equal-animal choice NLL (lower is better)')
    axes[0].set_title('Development prediction — exploratory')
    for i,m in enumerate(models):
        axes[0].text(means[m]+.001,i,f'{means[m]:.4f}',va='center',fontsize=9)
    axes[0].set_xlim(0,max(means.values())+.065)
    cells = list(decision['cells'])
    values = [decision['cells'][c] for c in cells]
    rates = np.array([c['detections']/c['runs'] for c in values])
    bounds = np.array([c['binomial_interval'] for c in values]).T
    for i,cell in enumerate(cells):
        injected = cell.startswith(('history_','mixed_'))
        axes[1].errorbar(rates[i],i,xerr=[[rates[i]-bounds[0,i]],[bounds[1,i]-rates[i]]],
                        fmt='o',color='#945332' if injected else '#386b8d',capsize=3)
    axes[1].plot([],[], 'o',color='#386b8d',label='No history injected: expect low')
    axes[1].plot([],[], 'o',color='#945332',label='History injected: expect high')
    axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16))
    axes[1].set_yticks(range(len(cells)),cells)
    axes[1].set_xlim(-.05,1.05)
    axes[1].set_xlabel('History detection fraction; exact 95% interval')
    axes[1].set_title('Recovery — simulated generating rules known')
    axes[1].axvline(.1,color='grey',linestyle=':',linewidth=1)
    axes[1].axvline(.8,color='grey',linestyle=':',linewidth=1)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig('docs/figures/generative_comparison_v1.png',dpi=180)
    fig.savefig('docs/figures/generative_comparison_v1.svg')
    plt.close(fig)
    lines = ['# Generative model comparison — September 19, 2026','',
             '**Decision: fresh confirmation remains deferred.**', '',
             'This comparison tests whether history adds prediction beyond explicit latent preference models. It does not establish a learning mechanism, neural necessity, controller superiority or novelty.', '',
             '[Protocol and simple mathematics](GENERATIVE_COMPARISON_PROTOCOL.md) · [Portable results](results/generative_comparison_v1.json)', '',
             '## Development results', '',
             f"Common sample: {development['summary']['trials']:,} trials from {development['summary']['animals']} development animals; five animal-disjoint folds.",'',
             '| Model | Equal-animal NLL |','|---|---:|']
    lines += [f'| {m} | {v:.6f} |' for m,v in sorted(means.items(),key=lambda x:x[1])]
    lines += ['', '| Added history model | Mean NLL gain | Clustered 95% interval |',
              '|---|---:|---|']
    for model,gain in development['summary']['gains'].items():
        lo,hi = gain['interval']
        lines.append(f"| {model} vs its no-history counterpart | {gain['mean']:.6f} | {lo:.6f}–{hi:.6f} |")
    lines += ['', 'These contrasts add the full history feature set. They are **not** estimates of the earlier difficulty-only residual gain of 0.000421 nats/trial. Models, nuisance features and scoring protocols must be compared within the same experiment.', '',
              '## Recovery and power', '',
              '| Generator / history strength | Oracle conditional KL | Detected / runs | Exact 95% interval | Selected fits converged | Qualification |',
              '|---|---:|---:|---|---|---|']
    for cell,v in decision['cells'].items():
        lo,hi=v['binomial_interval']
        lines.append(f"| {cell} | {v['mean_oracle_kl']:.6f} | {v['detections']}/{v['runs']} | {lo:.3f}–{hi:.3f} | {v['all_fits_converged']} | {v['pass']} |")
    lines += ['', f"Failed fits retained: {real_failed} development fits; {failed_fits} recovery fits. A failed optimizer prevents mechanistic interpretation of that run; its score is preserved, not silently discarded.", '',
              f"Individual optimizer attempts that hit a limit or failed: {real_failed_starts} development starts and {failed_starts} recovery starts. The table's convergence flag refers to the selected lowest-training-objective fit; all starts are retained in the raw artifacts. This does not certify a global optimum.", '',
              'The simulations reuse development task schedules and omission positions, with entirely new responses and lab/animal/session heterogeneity. Their conditional oracle effect size is reported in the portable results; it is not the detectable marginal gain. The observed winner frequencies are also retained. Forty repetitions per cell qualify only these specified scenarios, not universal power.', '',
              'A qualification failure is not proof that history and preference changes are always indistinguishable. In particular, one false detection out of 40 switching-only runs does not demonstrate a high false-positive rate; its confidence interval is too wide to certify the predefined 10% upper bound. Power and numerical resolution impose separate limits.', '',
              '## Verification and limitations', '',
              '- Independent reconstruction checks task/history features, common scoring rows, every saved development prediction, losses, summaries and source hashes.',
              '- The pinned author implementation checks GLM-HMM emission probabilities and causal filtering at the saved parameters. It does not independently refit models or reproduce the published paper.',
              '- A supplementary matched refit uses the author package likelihood and autograd derivatives, with the same features, priors, initialization strategy and scipy optimizer. This checks an independent likelihood implementation, not a reproduction of the published cohort or analysis.',
              '- The drift model is a fixed-parameter AR(1) grid approximation, not PsyTrack. A fitted continuous-drift benchmark remains required.',
              '- The supplementary 21-versus-41-state comparison checks numerical resolution without changing the primary grid.',
              '- These are previously inspected animals. Lab/animal bootstrap intervals are conditional on fitted models and omit uncertainty from refitting the training population.',
              '- Recovery uses two complementary animal folds while development uses five. These simulation results do not directly calibrate five-fold performance or the sample size for a future frozen-model confirmation study.',
              '- The simulator scales the entire history rule. It does not isolate a small difficulty interaction on top of strong ordinary history. Its power results must not be treated as a direct power calculation for the earlier difficulty-specific finding.',
              '- Synthetic dataframes are choice-analysis inputs, not complete schema-valid task rollouts; unused original fields such as reaction time are not simulated or consumed by the models.',
              '- Initial 250-iteration zero-initialized fits were interrupted after convergence failures; their plan, six completed recovery datasets, development scores and source snapshot remain in the original run directory. The second attempt uses training-GLM initialization and a 1000-iteration budget. This is a documented numerical revision, not a change to the scientific success thresholds.', '',
              '## What is next', '',
              'A new cohort requires reliable identification at the intended effect size, stable numerics, a fitted continuous-drift benchmark alongside the completed matched GLM-HMM refit, endpoint-specific power checks, and a new timestamped protocol with sample size and stopping rules. If recovery remains ambiguous, prioritize more informative within-animal recordings or a task manipulation with different predictions under drift and learning. Do not obtain fresh outcomes merely to repeat a favorable development score.', '',
              '![Development and recovery](figures/generative_comparison_v1.png)', '',
              'Reproduce with the commands in the protocol; all completed run artifacts are retained under `'+str(args.output)+'`. No new animal data were acquired.']
    resolution = checks['resolution_check.json']
    author = checks['author_refit.json']
    lines.extend(['', '## Numerical qualification results', '',
                  f"Drift grid qualification passed: **{resolution.get('pass', False)}**.", '',
                  f"Author-code matched refit qualification passed: **{author.get('pass', False)}**.", ''])
    for model, result in resolution.get('changes',{}).items():
        lines.append(f"- {model}: 41-minus-21-state NLL change {result['difference_41_minus_21']:.9f}; fixed tolerance 0.000100.")
    for model, difference in author.get('difference_from_primary',{}).items():
        lines.append(f"- {model}: independently refitted author-code NLL difference {difference:.9f}.")
    args.report.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    run(tyro.cli(Args))
