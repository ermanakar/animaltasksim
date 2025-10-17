# Understanding AnimalTaskSim: A High-Level Introduction

Welcome to AnimalTaskSim! This project sits at the exciting intersection of two major fields: **neuroscience** (the study of the brain) and **artificial intelligence** (the creation of intelligent agents).

At its heart, this project tackles a simple but profound question: **Can we create an AI that doesn't just win a game, but plays it like a real animal would?** This means capturing not just the successes, but also the biases, mistakes, and subtle patterns that are the telltale signs of a biological brain at work.

This guide will walk you through the core concepts without overwhelming you with technical jargon.

---

## 1. The Big Picture: Why Bother Replicating Animal Behavior?

For decades, AI has focused on creating "superhuman" agents—programs that can beat the world champion at chess or Go. This is incredibly impressive, but these AIs don't think like us. They often lack common sense and fail in ways that a human or animal never would.

Neuroscientists, on the other hand, study real brains. They have collected vast amounts of data showing how animals make decisions. They know animals are not perfect, optimal decision-makers. They are influenced by past events, they sometimes lose focus, and they take longer to decide when the choice is difficult.

**AnimalTaskSim is the bridge between these two worlds.** By forcing an AI agent to replicate the nuanced, sometimes "imperfect" behavior of an animal, we can test our theories about how the brain actually works. If we can build a model that acts like a mouse or a monkey, we might have discovered something fundamental about how their brains work.

### Why This Matters

- **For Neuroscience:** Validate theories about brain function by building working models
- **For AI:** Create agents that exhibit human-like reasoning, not just superhuman performance
- **For Medicine:** Understand decision-making deficits that appear in neurological conditions
- **For Science:** Bridge the gap between biological intelligence and artificial intelligence

---

## 2. The Core Concepts Explained

To understand the project, let's break down its three main pillars: **the Tasks**, **the Fingerprints**, and **the Models**.

### 2.1 The Tasks: Simple Games, Deep Insights

AnimalTaskSim currently supports two classic decision-making tasks from neuroscience:

#### Task 1: Mouse Visual Contrast Discrimination (IBL 2AFC)

The project focuses on a famous experiment from the **International Brain Laboratory (IBL)**. Think of it as a very simple video game:

- **Setup:** A mouse sits in front of a screen with a wheel it can turn left or right
- **Stimulus:** A visual grating (like a striped pattern) appears at varying contrast levels on either the left or right side
- **Goal:** The mouse must turn the wheel in the direction where the stimulus appeared to get a reward (a drop of water)
- **Difficulty:** Sometimes the stimulus is very clear and easy to see (high contrast). Other times, it's extremely faint and hard to distinguish (low contrast), making the choice difficult

```text
[Mouse 2AFC Task Diagram]

High Contrast (Easy):              Low Contrast (Hard):
┌──────────┬──────────┐            ┌──────────┬──────────┐
│          │ █████████│            │          │░░░░░░░░░░│
│          │ █████████│            │          │░░░░░░░░░░│
│          │ █████████│            │          │░░░░░░░░░░│
│  MOUSE   │  [WHEEL] │            │  MOUSE   │  [WHEEL] │
└──────────┴──────────┘            └──────────┴──────────┘
  Clear → Turn RIGHT                 Faint → Harder to tell!
      Reward: ✓                          Uncertainty high
```

#### Task 2: Macaque Random-Dot Motion (RDM)

This is another classic task from the Shadlen lab:

- **Setup:** A monkey watches dots moving on a screen
- **Stimulus:** Some percentage of dots move coherently in one direction (left or right), while others move randomly
- **Goal:** The monkey must judge the overall direction of motion
- **Difficulty:** When only a few dots move together (low coherence), it's very hard. When most dots move together (high coherence), it's easy

```text
[Random-Dot Motion Task Diagram]

High Coherence (Easy):        Low Coherence (Hard):
┌────────────────────┐        ┌────────────────────┐
│ → → → → → → → → →  │        │ → ↑ ← → ↓ ↗ ↖ ↘  │
│ → → → → → → → → →  │        │ ↓ → ↑ ← → ↗ ↓ ↖  │
│ → → → → → → → → →  │        │ → ← ↑ → ↘ ← → ↗  │
│ → → → → → → → → →  │        │ ↖ → ↓ ↑ → ← ↗ ↘  │
└────────────────────┘        └────────────────────┘
  90% dots moving RIGHT        30% coherent motion
  Decision: EASY ✓             Decision: HARD ⚠
```

Both tasks test the same fundamental question: **How do brains make decisions under uncertainty?**

This simple paradigm is powerful because it's controlled, repeatable, and generates rich data about decision-making.

---

### 2.2 The Fingerprints: What Makes Behavior "Animal-Like"?

When an animal performs these tasks thousands of times, it leaves behind a **"behavioral fingerprint"**—a set of unique patterns that reveal its cognitive strategy. An AI that just tries to get the most rewards won't show these patterns. AnimalTaskSim's goal is to build an agent whose fingerprint matches the real animal's.

Here are the key parts of the fingerprint:

#### 1. Psychometric Curve: Accuracy vs. Difficulty

This shows how the animal's accuracy changes as the task gets harder.

- **Easy trials** (high contrast/coherence): Nearly 100% correct
- **Hard trials** (low contrast/coherence): Performance drops toward chance (50/50)
- **The shape of this curve** reveals how sensitive the animal is to evidence

```text
[Psychometric Curve: Accuracy vs. Evidence Strength]

Accuracy
  100% ┤                 ╭────────
       │               ╭╯
       │             ╭╯
   75% ┤          ╭─╯
       │        ╭╯
       │      ╭╯
   50% ┤──────╯                    ← Chance level
       │
       └─────────┴─────────┴─────────┴──────
         0.0    0.25     0.5     1.0
              Evidence Strength
              (contrast/coherence)

Key: Steeper slope = more sensitive to evidence
     Flat at 50% = pure guessing at low evidence
```

#### 2. Chronometric Curve: Reaction Time vs. Difficulty

This shows how the animal's reaction time changes with difficulty.

- **Easy trials:** Quick decisions (low RT)
- **Hard trials:** Slower, more deliberate decisions (high RT)
- **This is crucial:** Animals instinctively take longer when uncertain

```text
[Chronometric Curve: Reaction Time vs. Evidence Strength]

Reaction
Time (ms)
       ╲
  900  ┤╲
       │ ╲
       │  ╲              ← Slow decisions when
  700  ┤   ╲                uncertain (low evidence)
       │    ╲
       │     ╲___
  500  ┤         ─────   ← Fast decisions when
       │              ─     confident (high evidence)
       │
  300  ┤
       └─────────┴─────────┴─────────┴──────
         0.0    0.25     0.5     1.0
              Evidence Strength

Key: Negative slope = evidence-dependent slowing
     This is the HARDEST fingerprint to replicate!
```

#### 3. History Effects: The Ghost of Trials Past

A purely rational agent would treat every trial as a new event. **Animals don't.**

Their choice on the current trial is subtly influenced by what happened on previous trials:

- **Win-Stay:** "I was rewarded for choosing left last time, so I have a slight bias to choose left again"
- **Lose-Shift:** "I was wrong when I chose right, so maybe I should try left this time"
- **Sticky Choice:** "I chose left three times in a row, so left feels more natural now"

These patterns reveal that animal brains use past experience to guide current decisions.

#### 4. Lapse Rate: Nobody's Perfect

Even on the easiest, most obvious trials, animals will occasionally make a silly mistake. This reflects:

- Momentary lapses in attention
- Fatigue or distraction
- Motor errors

**This is a key feature of biological intelligence**—perfect accuracy is impossible, even when the answer is obvious.

---

### 2.3 The Models: How Do We Build an Agent That Acts Like an Animal?

To reproduce such a complex fingerprint, this project uses a clever **hybrid model**, combining the strengths of a classic neuroscience model with a modern AI model.

#### Component 1: The Drift-Diffusion Model (DDM) — The Deliberator

**Concept:** Imagine you're deciding between two doors, Left and Right. You start in the middle. As you gather evidence for "Left," you drift closer to the Left door. As you gather evidence for "Right," you drift closer to the Right door. When you hit one of the doors, you've made your decision.

```text
[Drift-Diffusion Model: Evidence Accumulation Over Time]

Upper Boundary → CHOOSE RIGHT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                              ╱
                          ╱ ╱
                      ╱ ╱╲╱
                  ╱ ╱    
              ╱ ╱╲     
          ╱ ╱    ╲   ← Random walk
      ╱ ╱        ╲     (noise + drift)
Start ●────────────────────────────────
      │
      │ Bias ↓ (starting closer to one side)
      │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lower Boundary → CHOOSE LEFT

Time →
RT = Time to hit boundary + Non-decision time

Parameters:
• Drift rate (v): Speed toward correct boundary
• Boundary (a): Amount of evidence needed
• Bias (z): Starting point offset
• Noise (s): Random fluctuations
```

**Parameters:**

- **Drift rate:** How fast evidence accumulates (depends on stimulus strength)
- **Boundary separation:** How much evidence is needed before committing
- **Bias:** Starting point (closer to one boundary = preference for that choice)
- **Non-decision time:** Motor delays, stimulus encoding

**Strengths:**

- Brilliant at explaining the relationship between speed and accuracy
- The harder the decision (low drift rate), the longer it takes to hit a boundary
- Mathematically elegant and interpretable

**Weaknesses:**

- Too simple on its own
- No memory of past trials
- Can't capture history effects or attention lapses
- Parameters are fixed (doesn't adapt)

#### Component 2: The LSTM (Long Short-Term Memory) Network — The Learner

**Concept:** This is a powerful type of AI that excels at finding patterns in sequences of data. It has a "memory" that allows it to keep track of what happened in the past and use that information to inform the present.

```text
[LSTM: Sequential Memory and Learning]

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Trial N-3│   │ Trial N-2│   │ Trial N-1│   │ Trial N  │
│ contrast │   │ contrast │   │ contrast │   │ contrast │
│ choice   │──▶│ choice   │──▶│ choice   │──▶│ choice   │
│ reward   │   │ reward   │   │ reward   │   │ reward   │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────┐
│        LSTM Hidden State (Memory)                   │
│  [Tracks patterns: win-stay, lose-shift, etc.]     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │ DDM Parameters│
              │ • drift       │
              │ • bound       │
              │ • bias        │
              └──────────────┘
```

**Strengths:**

- Perfect for learning from the history of trials
- Can develop complex, adaptive strategies
- Captures sequential dependencies

**Weaknesses:**

- Can be a "black box"—internal reasoning is opaque
- Might learn to solve the task in biologically implausible ways
- No built-in structure for reaction times

#### Component 3: The Hybrid Solution — Best of Both Worlds

**AnimalTaskSim combines these two.** The LSTM acts as the **"coach"** and the DDM is the **"player."**

```text
[Hybrid DDM+LSTM Architecture: The Complete System]

┌──────────────────────────────────────────────────────────┐
│              LSTM "Coach" Network                        │
│                                                          │
│  Inputs:                                                 │
│    • Current stimulus (contrast/coherence)               │
│    • Previous 3 trials (choices, rewards, outcomes)      │
│    • Trial number / block context                        │
│                                                          │
│  Internal:                                               │
│    • Hidden state (256-dim memory)                       │
│    • Cell state (long-term patterns)                     │
│                                                          │
│  Outputs (per-trial DDM parameters):                     │
│    • drift_gain: How strongly to weight evidence         │
│    • bound: Decision threshold (speed/accuracy tradeoff) │
│    • bias: Starting preference (left/right)              │
│    • noise: Stochasticity level                          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       │ Parameters flow down ▼
                       │
┌──────────────────────┴───────────────────────────────────┐
│              DDM "Player" Simulator                      │
│                                                          │
│  Process:                                                │
│    1. Start at bias point                                │
│    2. Accumulate evidence: Δx = drift·dt + noise·√dt     │
│    3. Repeat until |x| ≥ bound OR timeout                │
│    4. Output choice (left/right) + RT                    │
│                                                          │
│  Why DDM?                                                │
│    • Produces realistic reaction times                   │
│    • Interpretable (we know WHY decision was made)       │
│    • Captures speed/accuracy tradeoff naturally          │
└──────────────────────────────────────────────────────────┘
                       │
                       │ Results flow to training ▼
                       │
┌──────────────────────┴───────────────────────────────────┐
│              Training with WFPT Loss                     │
│                                                          │
│  Loss = -log P(choice, RT | drift, bound, bias, noise)  │
│                                                          │
│  Gradients flow back to LSTM to improve parameter        │
│  predictions, making behavior more animal-like           │
└──────────────────────────────────────────────────────────┘

Key Insight: LSTM learns to set DDM parameters that
             reproduce animal behavioral fingerprints!
```

**How it works:**

On each trial, the LSTM (the coach) looks at:

- The current stimulus strength
- What happened on the last few trials
- The pattern of rewards and mistakes

Based on this, it gives instructions to the DDM (the player) by setting its parameters for that single trial:

- **"The last three choices were 'Left' and correct. Be a little biased towards 'Left' this time."** → Sets bias parameter
- **"This is a super easy trial. You can be confident and decide quickly."** → Sets lower boundary
- **"We haven't gotten a reward in a while. Be more cautious and take longer to decide."** → Sets higher boundary

**The result:**

- **Interpretable:** DDM structure means we can understand *why* the agent made each decision
- **Powerful:** LSTM learning means it can capture complex history effects
- **Biologically plausible:** Produces realistic reaction times and behavioral fingerprints

This hybrid approach creates a model that can reproduce the complex behavioral fingerprints of a real animal.

---

## 3. The Evaluation: How Do We Know If It Works?

AnimalTaskSim provides a rigorous evaluation framework:

### Step 1: Record Everything

Every trial generates a structured log entry containing:

- Stimulus parameters (contrast, coherence)
- Agent's choice (left/right)
- Reaction time
- Correctness and reward
- Previous trial outcomes
- Internal DDM parameters (drift, bound, bias)

**Format:** `.ndjson` (newline-delimited JSON) with schema validation

### Step 2: Compute Behavioral Metrics

From the trial logs, compute the fingerprints:

- **Psychometric curve:** Fit a sigmoid to accuracy vs. evidence
- **Chronometric curve:** Fit a line to RT vs. evidence
- **History effects:** Compute win-stay, lose-shift, sticky-choice rates
- **Bias and lapses:** Measure systematic biases and error rates

### Step 3: Compare to Reference Data

Generate side-by-side comparisons:

- **Agent behavioral fingerprints** (from your model)
- **Animal behavioral fingerprints** (from real experiments)

```text
[Dashboard Output: Agent vs. Animal Comparison]

╔════════════════════════════════════════════════════════╗
║          BEHAVIORAL FINGERPRINT COMPARISON             ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  Psychometric Curves:     Chronometric Curves:        ║
║  ┌──────────────┐         ┌──────────────┐           ║
║  │              │         │╲             │           ║
║  │      ╱───────│         │ ╲            │           ║
║  │    ╱         │         │  ╲___        │           ║
║  │  ╱           │         │      ───────│           ║
║  └──────────────┘         └──────────────┘           ║
║   • = Agent               • = Agent                   ║
║   × = Animal              × = Animal                  ║
║                                                        ║
╠════════════════════════════════════════════════════════╣
║  METRICS                  Agent    Animal   Match     ║
╠════════════════════════════════════════════════════════╣
║  Psychometric Slope       7.33     17.56    42%       ║
║  Chronometric Slope      -767     -645      84% ✓     ║
║  Win-Stay Rate           0.46     0.46     100% ✓     ║
║  Bias                    0.001    0.000    99% ✓      ║
╠════════════════════════════════════════════════════════╣
║  Overall Match Quality: 81% (GOOD)                    ║
╚════════════════════════════════════════════════════════╝

What this tells us:
✓ Chronometric match excellent (RT dynamics captured!)
⚠ Psychometric too shallow (agent underconfident)
✓ History effects match well
✓ Bias minimal (unbiased decision-making)
```

### Step 4: Iterate and Improve

- Adjust model architecture
- Tune hyperparameters
- Refine training curriculum
- Re-evaluate against reference data

**Goal:** Minimize the difference between agent and animal fingerprints

---

## 4. Putting It All Together: The Full Workflow

The `animaltasksim` repository provides the code to:

1. **Simulate:** Run the hybrid AI agent in virtual decision-making tasks
2. **Record:** Log all decisions, reaction times, and internal parameters with schema validation
3. **Analyze:** Compute psychometric, chronometric, and history metrics
4. **Compare:** Generate interactive HTML dashboards comparing agent vs. animal
5. **Refine:** Iterate on model design and training to improve behavioral match

By doing this, the project provides an invaluable tool for testing scientific hypotheses about the brain in a rigorous, reproducible way.

---

## 5. Key Insights & Scientific Value

### Why This Approach is Powerful

1. **Falsifiable:** If the agent can't match the fingerprints, the model is wrong
2. **Quantitative:** Metrics provide objective comparison (not subjective judgment)
3. **Reproducible:** Frozen schemas, deterministic seeds, saved configs
4. **Interpretable:** DDM structure reveals *why* decisions were made
5. **Extensible:** Framework generalizes to new tasks (PRL, DMS coming in v0.2)

### What We've Learned So Far

From the `FINDINGS.md` report:

- **Pure RL agents (PPO, Sticky-Q) fail at chronometry:** They produce flat RT curves—no evidence-dependent slowing
- **Hybrid DDM+LSTM succeeds:** First demonstration of an RL agent learning realistic chronometric slopes (-767 ms/unit vs. -645 ms/unit in macaques = 84% match)
- **Curriculum learning is critical:** Training must prioritize RT structure first, then choice accuracy
- **History effects are hard:** Win-stay/lose-shift patterns still underfit—more work needed

### Current Limitations (Honest Assessment)

- **Psychometric slopes too steep:** Agents are overconfident compared to animals
- **RT intercepts too slow:** Hybrid agent is ~500ms slower than macaques on average
- **History kernels underfit:** Sequential dependencies don't fully match animals
- **Limited to two tasks:** IBL 2AFC and RDM (PRL and DMS planned for v0.2)

**These limitations are documented transparently**—we're not hiding failures. Scientific progress requires honest reporting.

---

## 6. What's Next? How to Get Started

### For Users: Just Want to Run Experiments

```bash
# Interactive workflow (recommended)
python scripts/run_experiment.py

# Follow the wizard:
# 1. Select task (IBL Mouse or Macaque RDM)
# 2. Choose agent (PPO, Hybrid DDM+LSTM, etc.)
# 3. Configure parameters
# 4. Train → Evaluate → View results
```

**Explore existing results:**

- Browse `runs/` directory for experiment outputs
- Open `dashboard.html` files to see interactive comparisons
- Read `FINDINGS.md` to understand what we've learned

### For Researchers: Want to Test Your Data

```bash
# Convert your behavioral data to the unified schema
# See eval/schema_validator.py for format

# Compare your agent to your animals
python scripts/make_dashboard.py \
  --agent-log runs/my_agent/trials.ndjson \
  --reference-log data/my_animal/reference.ndjson \
  --output runs/my_agent/dashboard.html
```

**Query the experiment registry:**

```bash
# List all experiments
python scripts/query_registry.py list

# View detailed metrics for a specific run
python scripts/query_registry.py show --run-id my_experiment

# Export to CSV for analysis
python scripts/query_registry.py export --output experiments.csv
```

### For Developers: Want to Contribute

1. **Read the operating guide:** `AGENTS.md` contains implementation standards
2. **Check technical setup:** `README.md` for installation and workflows  
3. **Run tests:** `pytest tests/` (all 20 tests should pass)
4. **Follow the schema:** Use `eval/schema_validator.py` to ensure logs are compliant
5. **Adapt for new tasks:** Follow the Gymnasium interface in `envs/`

**Contribution areas:**

- New agent architectures
- Additional tasks (help with PRL/DMS!)
- Improved curriculum strategies
- Better history effect modeling

---

## 7. Further Reading

### Key Papers (All Cited in the Project)

**Drift-Diffusion Models:**

- Ratcliff & McKoon (2008). *Neural Computation*
- Ratcliff & Smith (2016). *Trends in Cognitive Sciences*

**Animal Data:**

- International Brain Laboratory (2021). *Neuron* — IBL mouse protocol
- Britten et al. (1992). *Journal of Neuroscience* — Macaque MT neurons
- Palmer, Huk & Shadlen (2005). *Journal of Vision* — RDM psychophysics

**History Effects:**

- Urai et al. (2019). *Nature Communications* — Choice history biases

### Project Documentation

- **Scientific validation:** `SCIENTIFIC_ASSESSMENT.md` — Independent review
- **Experimental results:** `FINDINGS.md` — What works, what doesn't
- **Developer guide:** `AGENTS.md` — Coding standards and workflows
- **Technical reference:** `README.md` — Installation, API, examples

---

## Glossary of Key Terms

**2AFC (Two-Alternative Forced Choice):** A task where the subject must choose between exactly two options

**Bias:** A systematic preference for one choice over another, even when evidence is equal

**Chronometric:** Relating to the measurement of time (in this context, reaction times)

**Coherence:** In RDM tasks, the percentage of dots moving in the same direction

**Contrast:** In visual tasks, the difference in brightness between stimulus and background

**DDM (Drift-Diffusion Model):** A mathematical model of decision-making as evidence accumulation

**Fingerprint:** The unique pattern of behaviors (accuracy, RT, history effects) that characterize a decision-maker

**History Effects:** The influence of past trials on current decisions

**Lapse:** An error made even on very easy trials, typically due to inattention

**LSTM (Long Short-Term Memory):** A type of recurrent neural network good at learning from sequences

**Psychometric:** Relating to the measurement of psychological functions (in this context, accuracy)

**RDM (Random-Dot Motion):** A task where subjects judge the direction of moving dots

**RT (Reaction Time):** The time between stimulus onset and response

**WFPT (Wiener First Passage Time):** The mathematical distribution of decision times in DDM

---

## Questions?

- **Technical support:** Open an issue on GitHub
- **Scientific inquiries:** See citations in `CITATION.cff`
- **Contributing:** Read `AGENTS.md` for guidelines

Welcome aboard! We're excited to have you explore the intersection of neuroscience and AI. 🧠🤖
