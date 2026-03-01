# AnimalTaskSim: Biological Decision-Making in AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/ermanakar/animaltasksim/actions/workflows/ci.yml/badge.svg)](https://github.com/ermanakar/animaltasksim/actions/workflows/ci.yml)

## Can an AI learn to think like a mouse?

Not just *win* like one — but **hesitate** on hard choices, **repeat** what worked last time, and **slow down** when the evidence is weak. Real brains don't just pick the right answer. They show specific patterns of speed, accuracy, habit, and error that reveal *how* they process information.

AnimalTaskSim puts AI agents into faithful recreations of real neuroscience experiments and asks: **does the agent produce the same behavioral patterns as the animal?** We call these patterns **behavioral fingerprints** — and matching them is much harder than maximizing reward.

> **Why does this matter?** If a computational model reproduces an animal's behavioral fingerprint, it becomes a testable theory of how that brain actually works. The model's architecture makes predictions about neural circuits that can be verified with real recordings.

---

## The Breakthrough: Three Bio-Inspired Mechanisms

When we first let an AI play these visual discrimination games, we ran into the **Decoupling Problem** — the moment the agent was allowed to learn both *visual evidence* and *history habits* simultaneously, it suffered from **mode collapse**. The AI realized that simply repeating its past actions was computationally easier than interpreting blurry visual evidence.

We solved this through three biologically-inspired mechanisms:

1. **Attention Gate**: When the stimulus is clear, the gate suppresses history bias; when it's ambiguous, history bias is allowed to influence the decision. This prevents mode collapse during joint learning.

2. **Asymmetric History Pathways**: Animals show a strong asymmetry — they repeat rewarded actions much more than they switch after errors (IBL mouse: win-stay=0.72 >> lose-shift=0.47, per-session means). We model this with separate win and lose networks that process reward and punishment through independent pathways, mirroring the dopaminergic asymmetry in the basal ganglia.

3. **Fixed Attentional Lapse**: Real animals occasionally err even on trivial trials (~5% of the time) due to momentary disengagement. We impose this as a fixed stochastic process — on a small fraction of trials, the agent guesses randomly. A learnable lapse parameter was tested and rejected: the optimizer exploited it as a shortcut, pushing lapse to ~15% to reduce choice loss on hard trials. Lapse in animals is a hardware property of the vigilance system, not a learned strategy.

---

## Results at a Glance

We tested our Hybrid DDM+LSTM agent on two classic tasks from decision neuroscience. After 70+ experiments, the agent simultaneously reproduces how animals *decide* (accuracy), how long they *deliberate* (reaction times), and how they're *influenced by the past* (history effects).

### IBL Mouse 2AFC — Current Best (5-seed validation)

| Metric | Agent (5-seed mean ± std) | IBL Reference (per-session) | Status |
|--------|--------------------------|----------------------------|--------|
| **Psychometric slope** | 12.38 ± 0.64 | 20.0 ± 5.7 | below reference mean |
| **Chronometric slope** | -34.2 ± 1.8 ms/unit | -51 ± 64 (lit. -36) | within reference range |
| **Win-stay** | 0.706 ± 0.008 | 0.72 ± 0.08 | within reference range |
| **Lose-shift** | 0.457 ± 0.007 | 0.47 ± 0.10 | within reference range |
| **Lapse rate** | ~0.075 | 0.08 ± 0.07 | within reference range |
| **Bias** | ~0.000 | ~0 | match |

> *5 seeds (42, 123, 456, 789, 1337), 3-phase curriculum, asymmetric history networks, 5% rollout lapse, co-evolution training with history injection (win_t=0.30, lose_t=0.15, drift_magnitude_target=9.0). Reference targets derived from per-session analysis of 10 IBL sessions (8,406 trials). History effects and chronometric slopes fall within the reference per-session range; psychometric slope is below the reference mean. See [FINDINGS.md](FINDINGS.md) for 70+ experiments, target provenance, and the full calibration narrative.*

### Macaque Random-Dot Motion — K2 Experiment

<p align="center">
  <img src="docs/figures/k2_dashboard.png" alt="Agent vs Macaque Behavioral Comparison" width="800">
</p>

| Metric | Agent | Macaque | Match |
|--------|-------|---------|-------|
| **Psychometric slope** | 10.7 | 17.6 | 61% |
| **Chronometric slope** | -270 ms/unit | -645 ms/unit | ✅ negative |
| **Choice bias** | 0.002 | ~0 | ✅ 99% |
| **Commit rate** | 100% | 100% | ✅ |

---

## The Three-Circuit Brain Architecture

Matching animal behavior required **separate computational pathways** that mimic actual brain organization:

```
                    ┌─────────────────────┐
  Current stimulus ─→   LSTM "Coach"       │──→ Base drift, boundary, bias
                    │   (evidence params)  │         │
                    └─────────────────────┘         │
                                                     ▼
                                          [ ATTENTION GATE ]
                                                     │
                    ┌─────────────────────┐          ▼
  Previous action  ─→   Win History MLP    │──→ ┌──────────────────┐    Choice
  Previous reward  ─→   Lose History MLP   │──→ │  Differentiable  │──→   +
                    │   (asymmetric stay)  │    │  DDM Simulator   │──→ Reaction
                    └─────────────────────┘    │  (Euler-Maruyama) │    Time
                                               └──────────────────┘
                    ┌─────────────────────┐          │
                    │  Fixed Lapse (5%)   │──────────┘
                    │  (stochastic gate)  │    On ~5% of trials,
                    └─────────────────────┘    agent guesses randomly
```

**Circuit 1 — "What do I see?"**: An LSTM learns to set Drift-Diffusion Model (DDM) parameters from stimulus features. The DDM accumulates evidence over time, producing slower responses on harder trials.

**Circuit 2 — "What worked last time?"**: Two separate MLPs process previous trial outcomes through independent win and lose pathways. After a reward, the win network computes a stay tendency; after an error, the lose network does. This asymmetry mirrors the dopaminergic split between reward and punishment processing in the basal ganglia, allowing the agent to learn win-stay and lose-shift at different rates — just as animals do (IBL mouse: win-stay=0.72 >> lose-shift=0.47, per-session means).

**Circuit 3 — "Am I paying attention?"**: On ~5% of trials, a stochastic lapse gate causes the agent to disengage and guess randomly. This models the attentional lapses observed in animals and is implemented as a fixed parameter, not a learnable one (see [FINDINGS.md](FINDINGS.md) for why).

**The DDM Simulator**: Rather than using analytical DDM equations (which create degenerate gradient landscapes), the training loop unrolls stochastic evidence accumulation as a differentiable PyTorch operation (Euler-Maruyama, 120 steps). This prevents the agent from exploiting mathematical loopholes to avoid learning.

> **Read more:** [Theory & Concepts Guide](docs/THEORY_AND_CONCEPTS.md) for an accessible deep dive, or [FINDINGS.md](FINDINGS.md) for the full 60+ experiment narrative.

---

## Immediate Roadmap

The Decoupling Problem is architecturally solved. Three bio-inspired circuits (evidence accumulation, asymmetric history, attentional lapse) simultaneously produce all six behavioral fingerprints, with history effects and chronometric slopes within the IBL per-session reference range. Psychometric slope remains below the reference mean. The active frontier:

1. **Learned history:** Current history effects use injected fixed values. Train the asymmetric win/lose networks to learn correct `stay_tendency` values from data (DDM-direct loss or RL-shaped reward).
2. **Psychometric slope gap:** Agent psych slope (12.38 ± 0.64) is below the per-session reference mean (20.0 ± 5.7). May require further drift calibration or gating changes.
3. **Lesion experiments:** Test architectural predictions — does removing the attention gate cause mode collapse? Does removing drift-rate bias eliminate history effects on easy trials?
4. **New cognitive tasks:** Probabilistic Reversal Learning (PRL), Delayed Match-to-Sample (DMS).
5. **Publication:** Three-circuit architecture + co-evolution calibration + negative results (60+ failed approaches).

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Interactive wizard (recommended) — walks you through everything
python scripts/run_experiment.py

# Or run manually:
python scripts/train_agent.py --env ibl_2afc --agent sticky_q --episodes 5 --seed 42 --out runs/my_run
python scripts/evaluate_agent.py --run runs/my_run
python scripts/make_dashboard.py \
  --opts.agent-log runs/my_run/trials.ndjson \
  --opts.reference-log data/ibl/reference.ndjson \
  --opts.output runs/my_run/dashboard.html
```

---

## Documentation

| Document | For whom | What's inside |
|----------|----------|---------------|
| [Theory & Concepts](docs/THEORY_AND_CONCEPTS.md) | Everyone | Accessible intro — tasks, fingerprints, how the model works |
| [FINDINGS.md](FINDINGS.md) | Researchers | 70+ experiments, what works, what fails, and why |
| [AGENTS.md](AGENTS.md) | Contributors | Implementation standards, workflow, coding guidelines |

---

## Reference Data

Benchmarked against two canonical datasets from decision neuroscience:
- **IBL Mouse 2AFC** — International Brain Laboratory (2021). 10 sessions, 8,406 trials. Standardized protocol across dozens of labs worldwide.
- **Macaque RDM** — Britten et al. (1992); Palmer, Huk & Shadlen (2005). 2,611 trials. The classic evidence accumulation paradigm.

---

## Acknowledgements

Developed independently by Erman Akar, with contributions from AI coding assistants (Claude, OpenAI Codex, Google Gemini).

**Scientific foundation:** Ratcliff & McKoon (2008); International Brain Laboratory (2021); Britten et al. (1992); Urai et al. (2019); Navarro & Fuss (2009).

**Built with:** Gymnasium, PyTorch, Stable-Baselines3, Pydantic, and the Scientific Python ecosystem.

---

## Citation

```bibtex
@software{akar2025animaltasksim,
  author = {Akar, Erman},
  title = {AnimalTaskSim: Hybrid Drift-Diffusion × LSTM Agents Matching Animal Decision Behavior},
  year = {2025},
  url = {https://github.com/ermanakar/animaltasksim},
  version = {0.1.0}
}
```

See [`CITATION.cff`](CITATION.cff) for additional citation metadata and references.
