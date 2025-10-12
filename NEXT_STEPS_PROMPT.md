# Agent Calibration: Next Steps

## Current Situation

We have successfully developed a structured, iterative workflow for calibrating the hybrid DDM+LSTM agent. Our most recent successful experiment, `runs/rt_calibration_v1/`, has produced an agent with a stable chronometric slope and improved psychometric and history metrics.

However, there is still a significant gap between the agent's behavior and the macaque reference data, particularly in the chronometric domain.

## Your Task

Your task is to continue the calibration process by following the structured workflow outlined in `CALIBRATION_GUIDE.md`.

Your specific goal is to implement and test a **per-coherence weighting schedule for the soft RT penalty**.

## Key Artifacts

- **Calibration Guide:** `CALIBRATION_GUIDE.md` - This is your primary reference for the workflow.
- **Latest Successful Run:** `runs/rt_calibration_v1/` - This is the baseline you should build upon.
- **Core Code Files:**
  - `agents/hybrid_ddm_lstm.py` (for curriculum and training loop modifications)
  - `agents/losses.py` (for loss function modifications)
  - `scripts/train_hybrid_curriculum.py` (for selecting the active curriculum)

## Recommended Plan

1. **Analyze the Latest Run:** Familiarize yourself with the results from `runs/rt_calibration_v1/dashboard.html` to understand the current state of the agent.
2. **Implement Per-Coherence Weighting:** Modify the training loop in `agents/hybrid_ddm_lstm.py` to construct and pass a `weights` tensor to the `soft_rt_penalty` function. This tensor should apply a stronger penalty to low-coherence trials and a weaker penalty to high-coherence trials.
3. **Create a New Curriculum:** Following the guide, create a new curriculum (e.g., `rt_weighted_calibration_curriculum`) that inherits from `rt_calibration_curriculum` and adds a new final phase to activate your new weighting scheme.
4. **Execute and Analyze:** Run the full scientific workflow (train, evaluate, dashboard) and analyze the results to determine if the per-coherence weighting has improved the agent's chronometric curve.
