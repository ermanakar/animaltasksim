# Agent Calibration Workflow: A Structured Approach

This document outlines the structured, iterative workflow for calibrating the hybrid DDM+LSTM agent's behavioral fingerprints to match the animal reference data. The core principle is to introduce one change at a time, test it rigorously, and analyze the results before proceeding.

## Guiding Principles

1. **One Variable at a Time:** Each new experiment should test a single, well-defined hypothesis (e.g., "increasing the choice loss will improve psychometric slope"). This is crucial for attributing changes in agent behavior to specific mechanisms.
2. **Build on Success:** Each new experiment should build upon the most successful previous curriculum. This ensures that we are always moving forward and preserving the progress we've made.
3. **Guardrails are Essential:** The per-phase guardrails (`min_slope`, `max_slope`, `max_sticky_choice`) are critical for ensuring that the agent's behavior remains within a scientifically plausible range. They provide a clear signal when a change has had an unintended negative consequence.

## The Calibration Workflow

The workflow is a three-step process: **1. Implement**, **2. Integrate**, and **3. Execute & Analyze**.

### Step 1: Implement the New Mechanism

Before running a new experiment, first implement the underlying code change. This typically involves:

- **Modifying Loss Functions:** Adjusting or adding new loss functions in `agents/losses.py`.
- **Adjusting the Training Loop:** Modifying the `train` method in `agents/hybrid_ddm_lstm.py` to incorporate new loss calculations or training procedures.

### Step 2: Integrate into a New Curriculum

Never modify an existing curriculum. Instead, create a new one to test your change.

1. **Create a New Curriculum Method:** In `agents/hybrid_ddm_lstm.py`, add a new static method to the `CurriculumConfig` class (e.g., `my_new_curriculum()`).
2. **Inherit and Extend:** The new curriculum should build on the previous best one (e.g., `base = CurriculumConfig.rt_calibration_curriculum()`).
3. **Add a New Phase:** Add a new, final phase to the `base.phases` list. This new phase should be the only one that activates your new mechanism (e.g., by setting a new loss weight). This isolates the change for clear analysis.
4. **Update the Training Script:** In `scripts/train_hybrid_curriculum.py`, change the default curriculum in the `main` function to your new one (e.g., `curriculum = CurriculumConfig.my_new_curriculum()`).

### Step 3: Execute and Analyze the Full Scientific Loop

This is the core iterative loop for testing your hypothesis.

1. **Train:** Run the training script with a new, descriptive output directory.

    ```bash
    python scripts/train_hybrid_curriculum.py --output_dir runs/my_experiment_v1
    ```

2. **Analyze the Training Run:**
    - Did all phases pass the guardrails?
    - If a phase failed, analyze the metrics in the console output to form a hypothesis about why (e.g., "The slope collapsed," "The agent became too sticky"). Adjust your mechanism or curriculum and return to Step 1.

3. **Evaluate:** If the training was successful, run the evaluation script to generate the final quantitative metrics.

    ```bash
    python scripts/evaluate_agent.py --run runs/my_experiment_v1
    ```

4. **Visualize:** Generate the dashboard for a comprehensive visual analysis.

    ```bash
    python scripts/make_dashboard.py --opts.agent-log runs/my_experiment_v1/trials.ndjson --opts.reference-log data/macaque/reference.ndjson --opts.output runs/my_experiment_v1/dashboard.html
    ```

5. **Hypothesize and Repeat:** Open the dashboard and analyze the results.
    - What is the biggest remaining discrepancy between the agent and the animal?
    - Form a new hypothesis for how to address it.
    - Return to Step 1 to implement and test your new hypothesis.

By following this structured process, we can ensure that our progress is steady, our results are interpretable, and our calibrations are scientifically sound.
