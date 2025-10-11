# AnimalTaskSim: AI Agents vs. Animal Behavior

## One-Page Summary for Quick Sharing

---

## What We Built

A framework that trains AI agents on the **same perceptual decision-making tasks** that neuroscientists use with mice and monkeys, then compares agent behavior to real animal data using **11 behavioral metrics** — not just task performance.

**Two benchmark tasks:**

- 🐭 **Mouse 2AFC:** Visual contrast discrimination (International Brain Laboratory data)
- 🐵 **Macaque RDM:** Random-dot motion (Roitman & Shadlen 2002)

---

## The Question

**Can reinforcement learning agents replicate animal-like behavior?**

Not just "get the task right," but exhibit the same:

- Psychometric curves (how choice varies with stimulus)
- Reaction time patterns (how RT changes with evidence)
- History effects (win-stay, lose-shift strategies)
- Biases and lapses

---

## The Answer

**Partially.** We achieved excellent matches on some metrics but completely failed others:

### ✅ What Held Up

- Task infrastructure stays modular and schema-validated, making it easy to compare log files and regenerate reports.
- History matches hover around chance for several agents, roughly aligning with published macaque win-stay ≈0.46.

### ❌ What Failed

- **Bias & RT claims need revision:** The latest Sticky-Q runs still sit ≈0.075 away from the mouse bias reference, and PPO re-checks remain far from macaque RT structure.
- **RT dynamics:** 0-22% match (agents make instant decisions, animals deliberate) even after disabling environment auto-commits.
- **History depth:** Only 79% match (agents too Markovian, animals consider multiple trials).
- **Lapses:** Agents at 37-49% vs animals ~0% (agents guess too much).

---

## Why This Matters

**The gap reveals fundamental differences between AI and biology:**

1. **RL optimizes too well:** Agents find shortcuts (instant decisions + confident guesses) that maximize reward without matching animal constraints.

2. **Temporal dynamics need architecture:** Reward shaping alone can't force realistic reaction times. Need mechanistic components like drift-diffusion models.

3. **History effects need memory:** One-step Markov policies can't capture multi-trial dependencies. Need recurrent connections (LSTM).

**Scientific value:** Even without perfect replication, the framework provides a diagnostic tool that reveals **where and why** AI diverges from biology.

---

## Example Results

### Mouse (Sticky-GLM v21)

```text
Psychometric slope: 18.50 vs 13.18 target (140% - too steep)
Bias: -0.001 vs 0.074 target (absolute gap ≈0.075)
Win-stay: 58.3% vs 73.4% target (79% ⚠️)
```

### Macaque (PPO recheck without collapsing bound)

```text
RT intercept: 60 ms vs 759 ms target (8% ❌)
RT slope: ~0 ms/unit vs -645 ms/unit target (0% ❌)
Win-stay: 56% vs 46% target (122% ❌ – agent over-stays wins)
```

### Macaque (DDM v2 - mechanistic baseline)

```text
RT intercept: 613 ms vs 759 ms target (81% ✅)
RT slope: -139 vs -645 ms/unit target (22% - correct direction! ✅)
```

**Key insight:** Mechanistic model (DDM) beats learned policy (PPO) on temporal dynamics because it has explicit evidence accumulation, not learned optimization.

---

## Interactive Dashboards

Open in your browser to explore:

- [`runs/ibl_final_dashboard.html`](runs/ibl_final_dashboard.html) - Mouse results
- [`runs/rdm_final_dashboard.html`](runs/rdm_final_dashboard.html) - Macaque PPO results  
- [`runs/rdm_ddm_dashboard.html`](runs/rdm_ddm_dashboard.html) - Macaque DDM results

Each shows:

- Side-by-side psychometric curves
- RT vs. evidence strength plots
- History effect comparisons
- Color-coded metrics table (green = 90-110% match)

---

## Path Forward: Hybrid Architecture

**Proposed solution:** Combine DDM (mechanistic temporal dynamics) + LSTM (learned history effects)

```text
Architecture:
1. LSTM: learns multi-trial dependencies → (drift, bound, bias) parameters
2. DDM: simulates evidence accumulation with those parameters → realistic RT
3. Combined: temporal realism + strategic learning
```

**Expected gains:**

- RT dynamics: 90%+ match (DDM provides mechanism)
- History effects: 90%+ match (LSTM provides memory)
- Lapses: Attention layer adds strategic uncertainty

**Estimated effort:** 6-9 weeks focused development

---

## Technical Stack

- **Python 3.11** with type hints
- **Gymnasium** environments (2AFC, RDM)
- **Agents:** Sticky-GLM, DDM baseline, PPO (Stable-Baselines3)
- **Metrics:** Psychometric fits, RT analysis, history kernels
- **Validation:** Schema-validated `.ndjson` logs, automated testing
- **CPU-only:** All training <2 hours on laptop

---

## Try It Yourself

```bash
# Train mouse agent
python scripts/calibration.py sticky --opts.task ibl --opts.n-episodes 25

# Train macaque agent  
python scripts/calibration.py ppo --opts.task rdm --opts.total-timesteps 250000

# Generate comparison dashboard
python scripts/make_dashboard.py \
  --opts.agent-log runs/my_run/trials.ndjson \
  --opts.reference-log data/ibl/reference.ndjson \
  --opts.output runs/my_dashboard.html
```

---

## Key Takeaway

**Behavioral replication ≠ reward optimization.**

RL agents are too smart, finding efficient shortcuts that animals (constrained by neurobiology) don't use. True replication requires:

- Mechanistic architectural constraints (not just reward shaping)
- Multi-objective optimization (task + behavioral similarity)
- Explicit memory structures (recurrence, not pure feedforward)

This work quantifies the gap and provides a testbed for closing it.

---

## More Information

- **Full analysis:** [FINDINGS.md](FINDINGS.md) (detailed results, technical insights, appendix)
- **Project docs:** [README.md](README.md) (usage, API reference)
- **Code:** [github.com/ermanakar/animaltasksim](https://github.com/ermanakar/animaltasksim)

---

**Version:** 0.1.0  
**Date:** October 2025  
**Status:** Ready for publication/sharing

---

## Citation

```text
AnimalTaskSim: A Benchmark for Evaluating Behavioral Replication in AI Agents
https://github.com/ermanakar/animaltasksim
October 2025
```
