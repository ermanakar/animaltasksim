# Understanding AnimalTaskSim

## The Elevator Pitch

Picture a mouse in a lab. A faint striped pattern flashes on a screen — left side or right side? The mouse spins a wheel to answer. Get it right, get a drop of water. Get it wrong, wait in the dark.

Now picture an AI agent playing the same game, with the same rules, the same timing, the same rewards. Here is the question we care about:

**Does the AI play the game the way the mouse does?**

Not just "does it get the right answer" — but does it hesitate on the hard trials the same way the mouse does? Does it repeat a choice after being rewarded, the way a real mouse would? Does it occasionally botch an easy trial because it just wasn't paying attention?

That is what AnimalTaskSim measures. It is a faithful digital copy of real neuroscience experiments, built to test whether an AI agent can reproduce the *specific patterns* of animal decision-making — the hesitations, the habits, and the mistakes, not just the correct answers.

---

## Why Build This?

Most AI research asks: *"How do we make the smartest agent possible?"*

We ask a different question: *"Can we build an agent that makes decisions the way a real brain does?"*

**Why that matters:**

- **For neuroscience.** If a computer model reproduces a mouse's behavior pattern-for-pattern, we may have discovered something about how that mouse's brain actually works. The model becomes a testable theory of the brain — you can "lesion" a component of the model (remove or disable it) and predict what would happen if you did the same to the real brain.

- **For AI.** Biological brains are not perfect calculators, but they are remarkably good at handling uncertainty, adapting to change, and making fast-enough decisions with noisy information. Understanding *how* they do it could inspire more robust AI systems.

The key insight is that animals are not optimal decision-makers. They are influenced by what happened on the last trial, they occasionally zone out, and they take longer to decide when the evidence is weak. These "imperfections" are not bugs — they are signatures of specific brain circuits doing specific computations. We call the full pattern of these signatures a **behavioral fingerprint**.

A guiding value runs through the whole project: **honest reporting over impressive results.** Past documentation once drifted into overstated claims and had to be corrected. Negative results — the things that did *not* work — are treated as first-class findings, because they narrow down what a real brain must be doing. You will see that ethos throughout this document: every claim is hedged to exactly what the evidence supports, and no further.

---

## The Tasks

AnimalTaskSim recreates two classic experiments from neuroscience and adds a third transfer probe (plus a fourth, memory-focused scaffold still under construction). Think of each one as a simple game with precise rules.

### Task 1: The Mouse Contrast Game (IBL 2AFC)

**The real experiment.** Developed by the [International Brain Laboratory](https://doi.org/10.1016/j.neuron.2021.04.001) and run in dozens of labs around the world. A mouse sits in front of a screen with a small steering wheel. A striped pattern appears on either the left or right side. The mouse spins the wheel to move the pattern to the center. Correct answer = a drop of water. Wrong answer = a brief wait in the dark.

**What makes it hard.** The pattern can be bold and obvious (high contrast, like black stripes on white) or so faint it is barely visible (low contrast, like light gray on slightly lighter gray). Low contrast trials are genuinely ambiguous — even the mouse's visual system cannot tell for sure which side the pattern is on.

**In the simulator.** The AI agent gets the same information the mouse would — a number representing how visible the pattern is and which side it appeared on — and must choose left or right. The trial timing, the reward rules, and even the way the lab shifts the odds between blocks of trials are all copied from the real protocol.

**Contrast levels used:** {0, 0.0625, 0.125, 0.25, 1.0} — five levels from invisible to unmistakable. (An earlier version of the simulator included a sixth level that does not appear in the real protocol; removing it improved the agent's accuracy pattern noticeably, with zero model changes — a good example of why protocol fidelity matters.)

This is the task the project has studied most thoroughly, and the one where the agent's behavioral fingerprint is validated.

### Task 2: The Monkey Dots Game (RDM)

**The real experiment.** Based on landmark studies from the [Shadlen lab](https://doi.org/10.1523/JNEUROSCI.12-12-04740.1992) that helped discover how neurons accumulate evidence during decisions. A rhesus macaque watches a cloud of moving dots. Some dots move together in the same direction (left or right), while the rest wander randomly. The monkey decides which way the majority of dots are moving.

**What makes it hard.** When most dots agree (high coherence), the answer is obvious. When only a few dots agree (low coherence), it looks like television static and the monkey has to watch carefully, accumulating evidence over time before committing to an answer.

**In the simulator.** The agent gets a coherence value (0% = pure noise, 100% = all dots agree) and must choose left or right. Importantly, the agent also controls *when* to respond — just like the real monkey, it has to decide when it has gathered enough evidence to commit.

**A caveat worth stating up front:** the macaque in the available reference dataset was overtrained and shows essentially no history effects. So RDM is useful for studying reaction-time dynamics, but it is *not* a good target for studying win-stay/lose-shift. Six months of history experiments were once run against this dataset before we realized the target simply did not exist — a lesson kept front-of-mind ever since.

### Task 3: The Silent Rule-Swap Game (PRL)

**The experiment.** Two choices look the same, but one pays out more often. The agent has to learn which one is better by trying them. Then, without warning, the payout probabilities swap.

**What makes it hard.** The environment does not announce the reversal. A single missed reward is ambiguous: the good option sometimes fails because the task is probabilistic. The agent has to accumulate evidence *from outcomes over time*, rather than react to a visible cue.

**In the simulator.** Both options stay visually neutral. The hidden payout contingencies are logged for offline analysis, but the acting agent only ever sees neutral options and the ordinary previous-trial outcome — it gets no oracle "the rule just flipped" flag. PRL is a *transfer probe*: an agent trained on the mouse task is dropped into this new task to see which of its decision circuits carry over. Crucially, **there is no PRL animal reference dataset in this repository**, so nothing here is an animal-parity claim; the findings are about which computational pieces are necessary inside the simulator.

### Task 4: The Memory Game (DMS) — scaffold only

Delayed Match-to-Sample: the agent sees a sample stimulus, waits through a delay, then judges whether a second stimulus matches. This probes *memory* rather than perception. Right now DMS is a schema-valid environment scaffold with a **defined** memory scorecard and lesion plan, but it is deliberately one step behind: its metrics, a memoryless baseline, and adaptive rollout are intentionally not wired yet. Building those controls comes before drawing any conclusions.

Together, the tasks ask: **How do brains decide under uncertainty, notice when the world changed, and remember what matters long enough to act?**

---

## The Scorecard: What Is a Behavioral Fingerprint?

We do not grade the AI the way you would grade a student ("what percentage did you get right?"). Instead, we look at **how it makes decisions** and check whether the patterns match what we see in real animals.

Five measurements, taken together, form the behavioral fingerprint:

### 1. The Accuracy Pattern (Psychometric Curve)

*"Does the agent get more accurate when the evidence gets stronger?"*

If you plot accuracy against evidence strength, you get a smooth S-shaped curve. Easy trials (strong evidence) produce near-perfect accuracy. Hard trials (weak evidence) produce near-random guessing. The steepness of the S-curve tells you how sensitive the decision-maker is — a steep curve means even small changes in evidence affect accuracy.

A real mouse has a specific steepness to its curve. Our agent should match it.

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#4CAF50"
---
xychart-beta
    title "Psychometric Curve"
    x-axis "Evidence Strength" [0, 0.125, 0.25, 0.5, 1.0]
    y-axis "Accuracy (%)" 45 --> 100
    line [50, 55, 65, 85, 98]
```

> A **steeper slope** means higher sensitivity to evidence. A flat line at 50% would mean the agent is guessing no matter what.

### 2. The Speed Pattern (Chronometric Curve)

*"Does the agent slow down when the evidence is weak?"*

This is the hardest fingerprint to reproduce. Real animals take measurably longer to respond on difficult trials. Their brains are literally accumulating noisy evidence over time — like trying to figure out which way a crowd is walking by watching one person at a time. When the crowd is mostly going left, you figure it out quickly. When it is split nearly 50/50, you have to watch a lot longer.

A standard AI just outputs an answer instantly. It has no concept of deliberation time. Getting an agent to naturally slow down on hard trials — without being explicitly told to — requires the right internal machinery.

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#FF9800"
---
xychart-beta
    title "Chronometric Curve"
    x-axis "Evidence Strength" [0, 0.125, 0.25, 0.5, 1.0]
    y-axis "Reaction Time (ms)" 400 --> 950
    line [900, 800, 680, 550, 480]
```

> A **negative slope** (slower for weak evidence, faster for strong evidence) is the hallmark of evidence accumulation — and the central challenge of this project.

> **A note on measuring reaction time.** "Reaction time" sounds unambiguous, but in the real IBL data you can measure it several ways — from stimulus onset to *first movement* of the wheel (~150 ms), or from stimulus onset to *response completion* (~400 ms). These give very different chronometric slopes. This project's calibrated targets use the response-completion convention, and getting that definition wrong quietly throws off every reaction-time comparison. It is a real footgun, discovered empirically (see FINDINGS.md, "IBL Reference Expansion").

### 3. History Effects (Win-Stay and Lose-Shift)

*"Is the agent influenced by what happened on the last trial?"*

A perfectly rational decision-maker would treat every trial as completely independent. Real animals do not. After a rewarded trial, a mouse tends to repeat its previous choice ("I went left and got water, so I will go left again"). After an unrewarded trial, it tends to switch ("I went right and got nothing, so let me try left").

These tendencies have names:

- **Win-stay:** The tendency to repeat a choice that was just rewarded. Think of it like a restaurant habit — you had a great meal at that Italian place, so you go back next Friday.
- **Lose-shift:** The tendency to switch away from a choice that was not rewarded. The sushi was bad, so you try Thai next time.

In mice, win-stay is much stronger than lose-shift (roughly 0.72 vs 0.47). This asymmetry reflects a real property of the dopamine system in the brain: reward signals are louder and more persistent than punishment signals.

### 4. Lapse Rate

*"Does the agent sometimes mess up on easy trials?"*

Even when the answer is completely obvious — a bold, high-contrast pattern on the left — a real mouse occasionally gets it wrong. Maybe it was grooming. Maybe it twitched. Maybe it just was not paying attention for a moment. These momentary disengagements happen on roughly 5-8% of trials.

A perfect AI would never do this. A biologically realistic AI should.

### 5. Bias and Commit Rate

*"Does the agent favor one side, and does it always give an answer?"*

A well-calibrated mouse has no strong left/right preference overall, and it responds on every trial (100% commit rate). Our agent should do the same.

---

## How the Agent's Brain Works

Standard reinforcement learning algorithms (like PPO, the kind used to train game-playing AIs) cannot produce realistic behavioral fingerprints. They respond instantly (no reaction times) and treat every trial independently (no history effects). We needed a different approach.

The solution has three independent circuits, each responsible for a different aspect of the behavioral fingerprint. This mirrors how real brains organize decision-making — different brain regions handle different parts of the process.

### Circuit 1: The Evidence Accumulator (DDM + LSTM)

**The core idea.** Imagine you are standing in a room during a rainstorm, trying to figure out if the wind is blowing left or right. Each raindrop that hits you gives a tiny clue — but each clue is noisy. Some drops splash left even when the wind is blowing right. So you wait, collecting evidence drop by drop, until you are confident enough to commit.

That is what a **Drift-Diffusion Model (DDM)** does. It is a mathematical model, originally developed to explain human and animal reaction times, that accumulates noisy evidence over time until it crosses a decision threshold.

The DDM naturally produces two key fingerprints:
- **The psychometric curve** — when more raindrops agree (strong evidence), the agent is more likely to get the right answer.
- **The chronometric curve** — when fewer raindrops agree (weak evidence), the agent takes longer to reach confidence.

But the DDM needs someone to set its dials: How sensitive should it be? How much evidence should it demand before committing? That is the job of the **LSTM** — a neural network that watches the stream of trials and adjusts the DDM's settings for each new trial. It outputs five parameters:

| DDM Setting | What It Controls | Brain Analogy |
|-------------|-----------------|---------------|
| Drift gain | How strongly evidence affects accumulation | Attention or gain modulation in sensory cortex |
| Decision bound | How much evidence is needed before committing | Speed-accuracy tradeoff (caudate/subthalamic nucleus) |
| Starting-point bias | Where accumulation begins (slight lean left or right) | Prior expectation (prefrontal cortex) |
| Noise level | How noisy the evidence stream is | Internal neural variability |
| Non-decision time | Motor delay before the response | Motor preparation time (motor cortex) |

### Circuit 2: The Habit System (Asymmetric History Networks)

**The problem.** In 70+ experiments, we found that the LSTM alone cannot produce history effects. Even when we explicitly trained it to reproduce win-stay and lose-shift behavior, the history signals it learned got drowned out during the DDM's evidence accumulation process. It is like trying to hear a whisper during a thunderstorm — the signal is there, but the noise of evidence accumulation washes it over.

**The solution.** Two small, separate neural networks — one for wins and one for losses — that bypass the LSTM entirely and directly nudge the DDM.

Why two networks instead of one? Because the brain handles rewards and punishments through different pathways. When a mouse gets a water reward, dopamine neurons in the ventral tegmental area fire strongly, creating a robust memory trace: "that action was good, do it again." When a mouse gets no reward, the signal is weaker and processed differently — through the lateral habenula and different dopamine circuits. The result: win-stay is much stronger than lose-shift in IBL mice. A single network processing both outcomes symmetrically cannot reproduce this asymmetry.

Each history network takes two inputs (what the agent chose last and whether it was rewarded) and outputs a single number: the **stay tendency** — how biased the agent should be toward repeating its previous choice.

The stay tendency reaches the DDM through two channels:

- **Starting-point bias.** Like leaning slightly toward one door before the evidence starts. This mostly affects ambiguous trials where the evidence is too weak to override the lean.
- **Drift-rate bias.** Like wearing tinted glasses that make left-favoring evidence look slightly stronger throughout the whole trial. This is the critical mechanism — it affects decisions at *all* difficulty levels, matching the observation that real mice show win-stay even when the stimulus is obvious.

An **attention gate** (`1 - |stimulus strength|`) controls how much the history bias is allowed through. When the stimulus is strong and obvious, the gate closes — sensory evidence dominates. When the stimulus is ambiguous, the gate opens — history has more influence. This prevents the history circuit from overwhelming the evidence circuit during training.

> **An honest caveat that recurs throughout this project:** the win-stay and lose-shift *strengths* are currently injected as hand-set numbers, not values the history networks discovered on their own. The architecture can *express* history effects faithfully; it cannot yet *learn* them from data. Teaching the networks to discover these tendencies is the single biggest open frontier.

### Circuit 3: The Attention Lapse

Sometimes, a mouse is just not paying attention. Maybe it is grooming, or resting, or momentarily distracted. On those trials, it effectively guesses randomly.

We model this as a simple coin flip: on about 5% of trials, the agent ignores all evidence and picks left or right at random. This is a fixed rate, not something the agent can learn to adjust.

Why fixed? Because we tried making it learnable, and the optimizer cheated — it pushed the lapse rate up to 15%, using random guessing as a shortcut to reduce loss on hard trials. That is not how real attention works. Attention lapses are a property of the biological hardware, not a strategy the brain optimizes.

### The Three Circuits Together

```mermaid
flowchart TD
    subgraph Inputs["What the agent sees"]
        S["Current stimulus\n(how strong is the evidence?)"]
        H["Last trial\n(what did I choose? was I rewarded?)"]
    end

    subgraph Circuit1["Circuit 1: Evidence Accumulator"]
        L["LSTM\nLearns patterns across trials"]
        DDM_params["DDM settings\n(drift, bound, bias, noise, delay)"]
    end

    subgraph Circuit2["Circuit 2: Habit System"]
        WIN["Win network\n(rewarded last trial)"]
        LOSE["Lose network\n(not rewarded last trial)"]
        GATE["Attention gate\nStrong stimulus = gate closes\nWeak stimulus = gate opens"]
    end

    subgraph DDM_sim["Differentiable DDM Simulator"]
        P["Accumulate noisy evidence\n120 steps, soft boundaries"]
    end

    subgraph Circuit3["Circuit 3: Attention"]
        LAPSE["Lapse gate\n~5% chance of random guess"]
    end

    subgraph Output["Trial output"]
        C["Choice (left / right)"]
        R["Reaction time (ms)"]
    end

    S --> L
    H --> L
    H --> WIN
    H --> LOSE
    L --> DDM_params
    WIN -->|"stay tendency"| GATE
    LOSE -->|"stay tendency"| GATE
    GATE -->|"bias on drift + start"| DDM_params
    DDM_params --> P
    P --> LAPSE
    LAPSE --> C
    LAPSE --> R
```

**A single trial, step by step:**

1. The agent sees a new stimulus (say, a faint pattern on the left).
2. The LSTM looks at the stimulus features and its memory of recent trials, then sets the DDM's base parameters.
3. The history network checks: "Last trial, I chose right and got rewarded." The *win* network fires, producing a positive stay tendency — a nudge toward repeating "right."
4. The attention gate checks: "The stimulus is faint (low contrast), so I'll let the history bias through." The DDM's drift rate and starting point both get nudged toward "right."
5. The DDM simulator runs: noisy evidence accumulates step by step. The stimulus says "left" but the history bias says "right." After 85 steps, the evidence crosses the "left" threshold. Choice: left. Reaction time: 85 steps in simulated time.
6. The lapse gate rolls its 5% die. Not a lapse trial this time, so the DDM's answer stands.
7. Output: left, 650 ms.

---

## How the Agent Learns

### Why Training Order Matters

You might think: "Just show the agent a bunch of mouse data and train on everything at once." We tried that. It does not work.

The reason is that the different objectives — matching reaction times, matching accuracy, matching history effects — compete with each other during training. If you push for better accuracy too early, the agent learns to crank its evidence sensitivity way up, which flattens the chronometric curve (it decides instantly regardless of difficulty). If you push for history effects before the evidence circuit is stable, the history bias overwhelms the stimulus signal and accuracy collapses.

The solution is a **3-phase curriculum**, inspired by how real brains develop — basic sensory processing first, then more complex decision-making layered on top:

```mermaid
flowchart LR
    A["Phase 1: Learn to deliberate\nRT training only\n(15 epochs)"] --> B["Phase 2: Learn to be accurate\nAdd choice loss\n(10 epochs)"]
    B --> C["Phase 3: Put it all together\nFull balance\n(10 epochs)"]
```

**Phase 1 — Reaction times only.** The agent learns the basic mechanics of evidence accumulation: slow down on hard trials, speed up on easy ones. No pressure on accuracy yet.

**Phase 2 — Add accuracy.** Now the agent also gets penalized for wrong answers. It learns to map stimulus strength to drift rate — stronger stimulus = stronger evidence signal = more accurate choices.

**Phase 3 — Full balance.** All objectives active together: accuracy, reaction times, history effects, and drift magnitude regularization (which prevents the DDM parameters from collapsing to extreme values).

(An earlier, more elaborate 7-phase curriculum actually made things *worse* — it let the optimizer crank up noise to game the loss, collapsing the accuracy pattern. Simpler won.)

### Co-Evolution: Evidence and History Must Grow Up Together

A critical discovery: you cannot train the evidence circuit first and bolt on history effects later. When the evidence circuit learns without history bias present, it calibrates itself for a world without interference. Adding history bias afterward throws everything off — accuracy drops because the evidence circuit was not built to handle the extra push.

The fix: **co-evolution training**. History injection is active from the start of training, so the evidence circuit learns to be stronger to compensate. This required recalibrating the drift magnitude target upward — the LSTM learns to push harder on the evidence signal because it "knows" that history bias will be pulling in its own direction.

This parallels a real phenomenon in brain development: sensory circuits and reward circuits have to mature together. If you disrupt one during development, the other does not compensate properly.

### The DDM Simulator: Why We Simulate Instead of Calculate

The DDM has well-known mathematical equations for its output distributions. Why not just use those equations during training?

Because the optimizer cheats. Given analytical DDM equations, the agent discovered it could push the decision boundary to infinity and the drift rate to zero. This zeroes out the reaction-time gradient (the math says "boundary is infinite, so all gradients vanish") while still maintaining okay choice accuracy through a mathematical shortcut. The result: the agent times out on every trial and learns nothing useful.

The fix: instead of using closed-form equations, we run the actual evidence accumulation process as a **differentiable simulation** — 120 steps of stochastic evidence accumulation, implemented in PyTorch so gradients flow through every step. The agent cannot exploit mathematical shortcuts because there are no shortcuts — it has to actually accumulate evidence the honest way.

---

## Current Results

The headline in plain language: **on the IBL mouse task, the hybrid DDM+LSTM agent reproduces all of the behavioral fingerprints at once** — the accuracy curve, the slow-down on hard trials, the win-stay/lose-shift asymmetry, the occasional lapse, and near-zero side bias with a 100% commit rate. Getting all of these *simultaneously* was the project's central challenge (the "Decoupling Problem"), and it is architecturally solved.

How good is the match? Across five random seeds, the history effects, the reaction-time slope, the lapse rate, and the bias all fall **within the range measured across real IBL mouse sessions**. The psychometric slope is the closest case: depending on the exact stimulus set and protocol details, it lands at or slightly below the reference mean. That single tightest metric is the one most sensitive to protocol details, which is why the project reports it carefully rather than rounding up.

Two honest qualifiers travel with this result:

- **History effects are injected, not learned** (as noted above). The agent *displays* realistic win-stay/lose-shift because those strengths are supplied by hand — the networks have not yet discovered them from data.
- **This is validated on IBL mouse only.** The macaque RDM task produces the right reaction-time dynamics but has no history effects to match (overtrained animal).

For the exact per-metric numbers, seed-by-seed spreads, and the full experimental history — including every result that later turned out to be an artifact — see **[FINDINGS.md](../FINDINGS.md)**. Those numbers are the source of truth and are kept current there; this document deliberately stays qualitative so it does not go stale.

### How Do We Know the Target Is Real? (Reference Expansion, July 2026)

A fair worry: what if the "mouse fingerprint" we are matching is just an artifact of ten hand-picked sessions? To test that, a fetcher script pulls quality-controlled sessions directly from the public IBL server. As of July 2026 it has independently reproduced **all six fingerprints on 80+ QC'd sessions** (tens of thousands of trials), once the correct reaction-time convention is used.

Two useful things came out of this. First, the reference fingerprint holds up on a much larger, independent sample — it is not an artifact of the small hand-curated set. Second, the wider sample revealed that real mice vary *more* between individuals than the small set suggested (especially in psychometric slope, where highly proficient mice have near-vertical curves) — which actually makes the agent's slope sit *more* comfortably within the natural range.

**Important:** this expanded data is **add-and-compare only**. The frozen 10-session reference remains the canonical target; nothing has been re-derived or adopted from the larger pull. Adopting a bigger reference would be a separate, deliberate decision. (See FINDINGS.md for the two infrastructure bugs this exercise caught — a session-selection bias and an inverted left/right action convention — both examples of "a bug in the measurement pipeline is a bug in the science.")

---

## Adaptive Control: Learning When to Persist, Switch, or Explore

The history-injection result above is the best *behavioral* match, but it does not solve learned history. So the research moved to a simpler, more biological question:

**Can an agent learn when to persist, switch, or explore after an uncertain outcome?**

The adaptive-control agent adds a small control system on top of the evidence accumulator. Loosely:

- the evidence core asks, "what does the stimulus say?"
- the outcome/value state asks, "what just happened?"
- a **persistence** pressure asks, "was that failure ambiguous enough that I should retry the same choice?"
- an **exploration** pressure asks, "is this context stale or uncertain enough that I should sample the alternative?"
- an **arbitration** step combines these without letting them erase strong sensory evidence.

The claim here is intentionally narrow, and the default is conservative.

**On the stable IBL task, `persistence_only` is the validated, default profile.** In a 5-seed matched lesion suite, an uncertainty-gated "retry after a weak/ambiguous failure" rule reliably reproduced an adaptive retry pattern, in all five seeds, while keeping exploration switched off. Exploration is *not* independently validated on stable IBL, so the full controller (persistence + exploration together) is reported only as a comparison condition, not the recommended default.

### The PRL Transfer Story (and a Reversal of Interpretation)

The interesting test is transfer: take that IBL-trained controller and drop it into the silent rule-swap task (PRL), where the payout odds flip without warning.

The first version of this experiment had a puzzle. The combined controller failed to adapt after reversals, and the *only* configuration that recovered was, oddly, "exploration-only." For a while the story was "exploration drives PRL learning."

A careful offline diagnostic overturned that story and found a simpler mechanism. In PRL, both options look visually neutral, so the agent's sensory-uncertainty dial (`1 - |stimulus|`) is pinned at its maximum on *every* trial. A retry rule that says "when uncertain and you just failed, try the same thing again" therefore fires at full strength after every single failure — producing relentless perseveration. "Exploration-only" wasn't winning because exploration is powerful; it was winning *by subtraction*, because it was the only lesion that happened to disable the misfiring retry rule.

**The fix: change-evidence recurrence (flag-gated, default OFF).** The single uncertainty signal is split into two dials:

- **Perceptual uncertainty** — the old sensory dial (`1 - |stimulus|`). Think: windshield visibility.
- **Change evidence** — a slow accumulator of recent failures that builds up as failures repeat and fades after wins. Think: a "you keep taking wrong turns" warning light.

When failures pile up, change evidence closes the retry gate and opens the switch gate — so the agent stops perseverating *without* needing to be told the rule flipped. In the stable IBL task, where failures don't pile up, this dial stays quiet and behavior is essentially unchanged. In fact, turning the flag off is a verified bit-for-bit no-op: it changes nothing unless you opt in.

A safety-gated calibration searched for how fast the accumulator should react. A fast setting (λ=0.7) was rejected as too jumpy — it would switch on ordinary bad luck. **λ=0.9 was selected as the validated opt-in cross-task profile.** With this recurrence active, the combined controller finally adapts after reversals, and on the IBL side it nearly preserves the original retry behavior.

**The interpretation reverses under the recurrence.** With the mechanism repaired, it is now *persistence*, not exploration, that drives PRL recovery — the exact opposite of the earlier "exploration wins" story, and for a principled reason (exploration was only ever winning by disabling a bug). Readers who saw the earlier framing should treat this as superseding it for the flag-on regime.

Three caveats bound all of this tightly:

- **This is not "solving PRL."** In absolute terms, these zero-shot agents perform only a little above chance. The claim is about *mechanism* — which computational piece is necessary — not about matching an animal.
- **There is no PRL animal reference**, so none of this is an animal-parity claim.
- **The feature stays default OFF.** λ=0.9 is an opt-in profile for explicitly labeled cross-task experiments; `persistence_only` remains the conservative standard default for the IBL task.

And a final honest note on the architecture itself: the "control system" is a computational analogy, lesion-tested in simulation, not a claim about specific brain anatomy. A follow-up lesion showed that even a simpler, linear uncertainty gate works *somewhat*, while the sharper nonlinear gate is stronger and more reliable — so the honest takeaway is "uncertainty-gated adaptive control is useful, and sharpening the gate helps," not "this exact gate shape is necessary."

---

## What Is Not Solved Yet

| Gap | Detail |
|-----|--------|
| History is injected, not learned | The win-stay and lose-shift strengths are hand-set numbers, not values the history networks discovered on their own. The architecture *can* express history effects, but it cannot yet *discover* them from data. This is the biggest open frontier. |
| Adaptive control is an analogy | The controller is lesion-tested in simulation, but it is not a neural-anatomy claim. |
| PRL animal parity is not tested | The lesion suite shows reproducible *in-simulator* mechanisms, but there is no PRL animal reference dataset in this repository, and absolute performance is near chance. |
| Combined adaptive control stays opt-in | The change-evidence recurrence repairs combined PRL recovery and λ=0.9 nearly preserves the IBL retry pattern, but it remains default off; `persistence_only` is the conservative standard IBL default. |
| DMS is a scaffold | The memory task's metrics, memoryless baseline, and adaptive rollout are defined but not yet wired. Controls come before conclusions. |
| Lapse variance across seeds | Lapse rates vary noticeably across seeds, suggesting the lapse mechanism interacts with training dynamics in ways we do not fully understand. |
| Single task validated | Behavioral-fingerprint results are validated on IBL mouse only. The macaque RDM task produces correct reaction-time dynamics but lacks history effects (overtrained animal). |

---

## Lessons Learned (The Hard Way)

Over 70+ experiments, we hit many critical failure modes. Here are the most instructive:

| What Went Wrong | Why | What Fixed It |
|----------------|-----|---------------|
| Agent always answered instantly, no matter the difficulty | Reinforcement learning (PPO) optimizes for reward, not deliberation | Replaced RL with a DDM that must accumulate evidence over time |
| Win-stay stuck at chance (0.50) for months | A bug: the agent never received reward information between trials | Fixed one line: read the reward at the right point in the trial phase cycle |
| Accuracy collapsed when we added a fancy training curriculum | The complex 7-phase curriculum let the optimizer crank up noise to game the loss function | Switched to a simpler 3-phase curriculum |
| Agent learned to "lapse" on 15% of trials as a strategy | A learnable lapse parameter was exploited by the optimizer — random guessing reduces loss on hard trials | Made lapse a fixed, non-learnable parameter (5%) |
| Six months optimizing for history effects on the wrong task | The macaque in the reference data was overtrained and had no history effects to match | Switched to IBL mouse data, which has robust history effects |
| Accuracy improved with zero model changes | The simulator included a contrast level that does not exist in the real experiment | Removed the bogus contrast level to match the actual protocol |
| An 84% "leftward bias" that wasn't real | The bias metric counted no-response trials in the denominator — a pipeline artifact, not agent behavior | Fixed the metric; always verify metrics reflect the agent, not the pipeline |
| "Exploration drives PRL learning" turned out to be a side effect | A misfiring retry rule under pinned uncertainty caused perseveration; exploration only "won" by disabling it | Split uncertainty into perceptual + change-evidence dials (change-evidence recurrence) |

The common theme: **most failures came from measurement or training-pipeline bugs, not from the architecture being wrong.** Getting the infrastructure right was as important as getting the model right — in this project, a bug in the measurement pipeline is a bug in the science.

---

## Getting Started

### Just Want to See It in Action?

```bash
pip install -e ".[dev]"
python scripts/run_experiment.py   # Interactive wizard
```

### Want to Train the Flagship Agent Directly?

```bash
python scripts/train_hybrid_curriculum.py \
    --task ibl_2afc --seed 42 --episodes 20 \
    --drift-scale 10.0 --drift-magnitude-target 9.0 \
    --lapse-rate 0.05 \
    --history-bias-scale 2.0 --history-drift-scale 0.3 \
    --inject-win-tendency 0.30 --inject-lose-tendency 0.15 \
    --no-use-default-curriculum --no-allow-early-stopping \
    --phase1-epochs 15 --phase2-epochs 10 --phase3-epochs 10
```

Training runs on CPU in under 20 minutes.

### Want to Dig Into the Results?

- Browse `runs/` for experiment outputs
- Open any `dashboard.html` to see agent vs. animal comparisons
- Read [FINDINGS.md](../FINDINGS.md) for the full experimental narrative (70+ experiments, including all the failures and exact numbers)

### Want to Contribute?

1. Read [AGENTS.md](../AGENTS.md) for coding standards
2. Run `pytest tests/` (176 tests should pass)
3. Areas where help is especially welcome: teaching the history networks to learn win-stay/lose-shift from data (the open frontier), lesion experiments, and implementing the DMS evaluator plus memoryless baseline from the defined fingerprint

---

## Glossary

| Term | Plain-Language Meaning |
|------|----------------------|
| **2AFC** | Two-Alternative Forced Choice — pick left or right, no other options |
| **Change evidence** | An adaptive-control dial (opt-in) that accumulates recent failures and fades after wins — a "you keep getting it wrong" signal, separate from sensory uncertainty |
| **Chronometric curve** | A plot of reaction time vs difficulty — shows whether the agent slows down on hard trials |
| **Co-evolution** | Training the evidence and history circuits together from the start, so they learn to work alongside each other |
| **Coherence** | In the monkey dots task: what fraction of dots move in the same direction (more = easier) |
| **Contrast** | In the mouse task: how visible the striped pattern is (higher = easier to see) |
| **DDM** | Drift-Diffusion Model — a mathematical model of how brains accumulate evidence before making a decision |
| **Decoupling Problem** | The challenge of getting one agent to simultaneously produce realistic reaction times AND history effects |
| **DMS** | Delayed Match-to-Sample — a memory task (see a sample, wait, judge whether a second stimulus matches). A scaffold here, not yet wired |
| **Drift-rate bias** | A history-driven nudge that affects how evidence is processed throughout the whole trial, not just at the start |
| **Fingerprint** | The full pattern of accuracy, speed, history, and lapses that characterizes a particular decision-maker |
| **History effects** | How the outcome of the previous trial influences the current decision (win-stay, lose-shift) |
| **Lapse** | A trial where the decision-maker makes an error despite strong evidence — usually due to momentary inattention |
| **LSTM** | Long Short-Term Memory network — a type of neural network that remembers patterns across a sequence of events |
| **Perceptual uncertainty** | The sensory "how ambiguous is the stimulus?" dial (`1 - |stimulus|`) — pinned at maximum in PRL, which is why the retry rule misfired there |
| **Persistence-only** | The validated, default adaptive-control profile on IBL: retry after ambiguous failures, with exploration switched off |
| **Psychometric curve** | A plot of accuracy vs evidence strength — shows how sensitive the decision-maker is |
| **PRL** | Probabilistic Reversal Learning — choose between neutral options, then adapt when their hidden payout odds silently swap |
| **RDM** | Random-Dot Motion — the monkey dot-direction task |
| **RT** | Reaction Time — how long it takes to respond (note: the *definition* of RT matters; see the chronometric section) |
| **Stay tendency** | How biased the agent is toward repeating its previous choice |

---

## Further Reading

**The Science Behind the Project:**
- Ratcliff & McKoon (2008). *Neural Computation* — The mathematical foundations of drift-diffusion models
- International Brain Laboratory (2021). *Neuron* — The standardized mouse decision-making task
- Britten et al. (1992). *Journal of Neuroscience* — The classic monkey dot-motion experiment
- Urai et al. (2019). *Nature Communications* — How past outcomes bias future decisions in animals

**Project Documentation:**
- [FINDINGS.md](../FINDINGS.md) — The full experimental narrative and exact, current numbers (70+ experiments, all failures documented)
- [AGENTS.md](../AGENTS.md) — Developer guide and coding standards
- [README.md](../README.md) — Installation and quickstart
- [DMS Memory Fingerprint Design](dms_memory_fingerprint_design.md) — Memory-task scorecard and rollout prerequisites
