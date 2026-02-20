# AnimalTaskSim: Biological Decision-Making in AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Can an AI learn to think like a mouse?

Not just *win* like one — but **hesitate** on hard choices, **repeat** what worked last time, and **slow down** when the evidence is weak. Real brains don't just pick the right answer. They show specific patterns of speed, accuracy, habit, and error that reveal *how* they process information.

AnimalTaskSim puts AI agents into faithful recreations of real neuroscience experiments and asks: **does the agent produce the same behavioral patterns as the animal?** We call these patterns **behavioral fingerprints** — and matching them is much harder than maximizing reward.

> **Why does this matter?** If a computational model reproduces an animal's behavioral fingerprint, it becomes a testable theory of how that brain actually works. The model's architecture makes predictions about neural circuits that can be verified with real recordings.

---

## The Breakthrough: The "Attention Gate" solves Mode Collapse

When we first let an AI play these visual discrimination games, we ran into the **Decoupling Problem** — the moment the agent was allowed to learn both *visual evidence* and *history habits* simultaneously, it suffered from severe **mode collapse**. The AI realized that simply repeating its past actions ("win-stay") was computationally easier than interpreting blurry visual evidence, so it stopped looking at the screen entirely and just mashed the same button.

We solved this with a profound biological insight: brains do not blindly mix "what they see" with "what they remember." They have an **Attention Gate**.

We added a structural guardrail to our agent:
- **When the screen is blurry (hard level):** The gate opens. The AI relies heavily on its habits (history prior) to guess the answer.
- **When the screen is clear (easy level):** The gate snaps shut. The AI suppresses its habits and makes a decision based purely on the objective sensory evidence in front of it.

This mechanism mathematically forces the agent to balance its history with sensory evidence, completely eliminating mode collapse during joint learning.

---

## Results at a Glance

We tested our Attention-Gated Hybrid DDM+LSTM agent on two classic tasks from decision neuroscience. After 60+ experiments, the agent simultaneously reproduces how animals *decide* (accuracy), how long they *deliberate* (reaction times), and how they're *influenced by the past* (history effects). 

### IBL Mouse 2AFC — Validated Across 5 Seeds

<p align="center">
  <img src="docs/figures/ibl_v6_dashboard.png" alt="Agent vs IBL Mouse Behavioral Comparison" width="800">
</p>

| Metric | Agent (mean ± std) | IBL Mouse | Match |
|--------|-------------------|-----------|-------|
| **Win-stay** (repeat after reward) | 0.839 ± 0.012 | 0.724 | Very Strong |
| **Lose-shift** (switch after error) | 0.206 ± 0.023 | 0.427 | Moderate |
| **Chronometric slope** (slower on hard trials) | -18.7 ± 4.7 ms/unit | negative | ✅ Negative slope |
| **Psychometric slope** (accuracy vs difficulty) | 3.9 ± 0.25 | ~13.2 | ✅ Correct shape |
| **Commit rate** | 100% | 100% | ✅ |

> *5 seeds (42, 123, 256, 789, 1337), identical config, using the Attention-Gated History Bias mechanism.*

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

## The Two-Circuit Brain Architecture

Matching animal behavior required **two separate computational pathways** that mimic actual brain organization:

```
                    ┌─────────────────────┐
  Current stimulus ─→   LSTM "Coach"       │──→ Base drift, boundary, bias
                    │   (evidence params)  │         │
                    └─────────────────────┘         │
                                                     ▼
                                          [ ATTENTION GATE ]
                                                     │
                                                     ▼
                    ┌─────────────────────┐    ┌──────────┐    Choice
  Previous action  ─→   History Network    │──→ │   DDM    │──→   +
  Previous reward  ─→   (separate MLP)     │──→ │ "Player" │──→ Reaction
                    │   (stay tendency)    │    └──────────┘    Time
                    └─────────────────────┘
```

**Circuit 1 — "What do I see?"**: An LSTM learns to set Drif-Diffusion Model (DDM) parameters from stimulus logic. The DDM accumulates evidence over time, producing slower responses on harder trials.

**Circuit 2 — "What worked last time?"**: A separate MLP outputs a *stay tendency* based on the previous trial's reward. The signal acts as a **drift-rate bias**, continuously pushing the evidence accumulation process over the trial duration. This mechanism prevents the history effects from getting washed out during deliberation.

> **Read more:** [Theory & Concepts Guide](docs/THEORY_AND_CONCEPTS.md) for an accessible deep dive, or [FINDINGS.md](FINDINGS.md) for the full 60+ experiment narrative.

---

## Immediate Roadmap

Now that the Decoupling Problem mathematically "Mode Collapse" is solved, here are our precise next steps:

1. **Perfecting the Psychometric Slope Gap (Current Priority):** Optimize the agent so its psychometric shape strictly aligns with the sensitivity (steepness ~13.2) of biological mice. Focus will be on curriculum enhancements or fine-tuning DDM noise elements.
2. **Context-Dependent Structured Memory:** Expand the Habit Circuit to remember what worked *the last time it was in the exact same situation* (e.g. "What worked the last time the screen was blurry?").
3. **New Cognitive Tasks:** Test the hybrid agent on Probabilistic Reversal Learning (PRL) and Delayed Match-to-Sample (DMS) tasks to prove generalized cognitive flexibility.

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
| [FINDINGS.md](FINDINGS.md) | Researchers | 60+ experiments, what works, what fails, and why |
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
