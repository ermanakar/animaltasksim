# AnimalTaskSim Findings

**Benchmarking reinforcement learning agents against rodent and primate decision-making fingerprints**  
*October 2025 · Version 0.1.0*

---

## Executive Summary

AnimalTaskSim measures how closely AI agents reproduce animal behavioral fingerprints on two classic tasks. Baseline agents match several bias and history statistics but diverge on reaction-time dynamics and lapse patterns. Architectural inductive biases, not just reward shaping, remain the limiting factor.

- Sticky-GLM (IBL 2AFC): nails contrast bias (99% match) yet under-expresses win-stay (79%) and overproduces high-contrast lapses (731%).
- PPO (RDM): maximizes reward but collapses RT structure (28% intercept match) and introduces large directional bias (0.247 vs ~0).
- Drift Diffusion Model (RDM): best RT alignment (81% intercept match) at the cost of overly shallow psychometric slopes (37%).

Dashboards for each run live under `runs/` and pair visuals with metric deltas.

---

## Task Highlights

### Mouse Visual 2AFC (IBL)
- **Reference benchmarks** (885 trials): slope 13.2, bias 0.074, win-stay 73%, lose-shift 34%, sticky-choice 72%, lapse_high 0.0026.
- **Sticky-GLM v21** (25 training episodes): slope 18.5 (140%), bias -0.001 (99%), win-stay 58% (79%), lose-shift 34% (99%), sticky-choice 57% (79%), lapse_high 0.019 (731%).
- **Interpretation:** Linear policy reproduces additive biases but lacks depth in multi-trial history and over-fits stimulus features, yielding excess confident errors on easy trials.

### Macaque Random-Dot Motion (RDM)
- **Reference benchmarks** (2611 trials): slope 17.6, bias ~0, RT intercept 759 ms, RT slope -645 ms, win-stay 46%, sticky-choice 46%.
- **PPO v24:** slope 30.4 (173%), bias 0.247, RT intercept 210 ms (28%), RT slope 0 ms (0%), win-stay 46% (100%).  *Fast, reward-optimal, but non-biological RT curve and large choice bias.*
- **DDM v2:** slope 6.5 (37%), bias 0.015, RT intercept 613 ms (81%), RT slope -139 ms (22%), win-stay 53% (116%).  *Captures temporal dynamics via evidence accumulation but misses accuracy scaling.*

---

## Key Insights

1. **Reward-optimal ≠ behaviorally faithful.** Agents maximizing reward alone diverge on RT and lapse structure found in animal data.
2. **Architectural priors matter.** Mechanistic models (DDM) inherit realistic temporal dynamics; generic policy gradients do not, even with history features.
3. **Multi-trial memory remains an open gap.** Sticky-GLM and PPO capture first-order history but underperform on deeper kernels observed in rodents and primates.
4. **Schema-first logging pays off.** Shared `.ndjson` format lets metrics, dashboards, and future tasks plug in without interface churn.

---

## Next Steps

1. Hybrid agent combining DDM-style accumulation with adaptive policy layers to balance RT realism and accuracy.
2. History-regularized training objectives to encourage biologically plausible win-stay/lose-shift patterns.
3. Extend the pipeline to PRL and DMS while reusing seeds, schema validation, and reporting hooks.

---

## Artifacts

- Mouse dashboard: `runs/ibl_final_dashboard.html`
- Macaque PPO dashboard: `runs/rdm_final_dashboard.html`
- Macaque DDM dashboard: `runs/rdm_ddm_dashboard.html`
- Metrics JSON and config snapshots live alongside each run directory.

---

## Citation

```
AnimalTaskSim: A Benchmark for Evaluating Behavioral Replication in AI Agents
https://github.com/ermanakar/animaltasksim
October 2025
```
