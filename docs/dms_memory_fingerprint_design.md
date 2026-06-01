# Design Note: DMS Memory Fingerprint Before Adaptive Rollout

**Status:** FINGERPRINT DEFINED, implementation intentionally limited to the
schema-valid environment scaffold.

**Date:** June 1, 2026

## 1. Why DMS Is Next

Delayed Match-to-Sample (DMS) tests short-term memory. The agent sees a sample,
waits through a silent delay, sees a test stimulus, then reports whether the two
match.

The plain-English analogy is a card game: show someone a card, hide it behind a
curtain, then show a second card. The question is not whether they can see the
second card. It is whether they kept the first card in mind long enough to
compare the two.

That is scientifically different from IBL and PRL:

- IBL asks whether sensory evidence and recent outcomes shape a perceptual choice.
- PRL asks whether repeated failures reveal that a hidden payout rule changed.
- DMS asks whether information survives a quiet delay and can be retrieved for
  comparison.

The adaptive-control rollout stays unwired until the memory fingerprint is
defined and measured. Otherwise a high score could come from a shortcut rather
than memory.

## 2. Existing Scaffold

`envs/dms_match.py` already provides a schema-valid environment:

```text
iti -> sample -> delay -> test -> response -> outcome
```

The sample and test use the same absolute contrast magnitude. On non-match
trials, the side flips. This prevents the agent from solving the task by
noticing that one stimulus is stronger than the other.

The frozen `.ndjson` contract already carries the fields needed for offline
analysis:

- `sample_stimulus`
- `stimulus` for the test stimulus
- `delay_ms`
- `match`
- `action`, `correct`, `reward`, `rt_ms`, and `phase_times`

The acting policy must not receive `match` as an observation. It is an
offline-analysis label only.

## 3. Memory-Specific Scorecard

The first DMS evaluator should report:

| Metric | Why it matters |
|---|---|
| Overall committed accuracy | Basic task performance; chance is `0.5` |
| Accuracy by `delay_ms` | Retention curve: memory should weaken as the delay grows |
| Accuracy by sample/test magnitude | Separates weak encoding from memory loss |
| Match vs non-match accuracy | Detects an always-answer-match or always-answer-non-match shortcut |
| Match-choice bias | Detects response imbalance even when total accuracy looks acceptable |
| Commit rate and omissions | Prevents selective non-response from inflating accuracy |
| RT by delay and magnitude | Secondary retrieval-cost fingerprint |
| Previous-outcome carryover | Secondary control: DMS memory should not reduce to sticky reward history |

The scaffold currently uses one fixed delay. A future experiment may add an
additive delay schedule, but it must preserve existing CLI and schema fields.

## 4. Lesion Ladder

The first claim-bearing DMS run needs matched controls:

| Condition | Expected role |
|---|---|
| Memoryless current-test-only baseline | Must stay near chance on balanced match/non-match trials |
| Recurrent memory agent | Should exceed the memoryless baseline and remain above chance |
| Delay-state reset lesion | Reset memory at the delay boundary; performance should collapse toward the memoryless baseline |
| Sample-visible oracle diagnostic | Sanity-check ceiling only; never a claim-bearing agent |

The key causal comparison is recurrent memory versus the delay-state reset
lesion. That is the equivalent of closing the curtain and checking whether the
agent was genuinely remembering the card.

## 5. Promotion Criteria

Before wiring adaptive-control DMS rollout:

1. Add DMS metrics to the shared evaluator without changing frozen schema keys.
2. Add a memoryless baseline and prove it stays near chance.
3. Add a variable-delay protocol and confirm a sensible retention curve.
4. Show recurrent memory beats the memoryless and delay-reset lesions.
5. Confirm match/non-match balance, schema validation, deterministic seeding,
   and CPU-friendly runtime.

## 6. Claim Boundary

This note defines the memory fingerprint. It does not implement an adaptive
controller for DMS, claim animal parity, or change the trial schema.

The next engineering milestone is the evaluator plus memoryless baseline. Only
after those controls are green should adaptive-control rollout be connected.
