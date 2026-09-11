# Architecture repair — September 2026

This repair concerns the Hybrid/DDM/adaptive architecture. It does not alter the
completed frozen animal-cohort replication, its models, protocol, or findings.
Historical runs are retained as records; they are not results of repaired code.

## Interpretation changes

- PRL rollout supplied previous optimal-option correctness to the neural core,
  although the agent was supposed to infer hidden contingencies from stochastic
  reward. Historical PRL numbers cannot establish reward-only transfer. Removing
  that information requires new experiments; its quantitative contribution is
  not yet known.
- The standard validation suite trains a separate model per profile and seed.
  These are retrained profile ablations, not acute lesions of one trained model.
  Shared seed numbers do not separate training compensation from rollout effects.
- Full-control PRL block-learning lift partly reflects poorer early post-reversal
  performance. Compare overall optimal choice, reversal cost, recovery trajectory,
  and late-block performance separately. Large late-minus-early differences alone
  do not establish better adaptation.
- Historical injected history tendencies and fixed lapse probabilities are
  calibrated ingredients, not evidence that the network independently learned
  those fingerprints. Comparison to simple DDM/history and reversal-learning
  baselines remains necessary.

## Reproducible retrained suites

The suite records `suite_provenance.json` only after successful training and
analysis. It binds exact CLI commands, Python version, training reference bytes,
Python source bytes, and hashes of all five required run artifacts. Existing
runs can be reused only when these match. Missing manifests, modified outputs,
changed source/reference, and partial runs fail closed. Use a fresh `run_root`;
even disabling `skip_existing` does not authorize overwriting historical runs.

This protection applies to the retrained validation suite and its subclasses.
Legacy standalone sweep/diagnostic scripts are not automatically provenance-safe.
Package/environment reproducibility beyond the recorded Python version still
requires preserving the environment; code and data hashes alone do not provide it.

## What a decisive architectural experiment still requires

1. Freeze one trained full-control checkpoint and evaluate runtime interventions
   with the same allowed observations and controlled trial randomness.
2. Run separate retrained ablations to measure compensation during learning.
3. Include simple baselines with the same sensory/reward information and budget.
4. Score joint choices and reaction-time distributions, omissions, and reversal
   trajectories rather than selecting only favorable summary metrics.
5. Keep all repaired results in new directories, with explicit provenance and
   bounded claims. Software checks establish implementation behavior, not an
   animal mechanism or scientific novelty.

## Acute checkpoint comparisons

`scripts/checkpoint_ablation.py` evaluates all four runtime profiles from the
same strict-loaded weights, without calling training. It requires an empty new
output root, checks parameter equality after each rollout, and hashes source
checkpoint/config and each output. A reference-path override supports moved
checkpoints; model initialization reads that reference but does not optimize.
The manifest identifies these results as fixed-checkpoint runtime interventions.

The optional rollout pairing uses separate trial-indexed lapse and DDM streams.
Each trial receives the same noise prefix and lapse draws across profiles,
independent of preceding response duration. Environment seeds are shared; reward
outcomes can differ because profiles choose different actions. These comparisons
measure acute effects conditional on the source checkpoint, not compensation
from training or biological necessity. Evaluating a historical checkpoint under
repaired code is a new evaluation, not an unchanged replay of its old semantics.

## Training and decision-path repairs

The repair carries detached forward state through chronological chunks within
shuffled sessions, uses full-session time normalization, differentiates expected
conditional history rates, derives RT targets from fitting data, and gives the
twin evidence-gain pass the actual stimulus. Lapse mixture and timeout handling
are aligned across training and rollout. The differentiable first-passage model
remains a soft-boundary approximation; passing regression tests does not show
that its gradients reproduce an exact DDM likelihood.

## Numerical calibration and remaining approximation

The old smoothing width 0.1 anticipated hard boundary crossings by 106–223 ms
in three seeded checks (3,000 paired paths per setting). The new default width
0.01 reduced those mean RT discrepancies to 2.3–3.6 ms; maximum choice-probability
discrepancy was approximately 0.00035. Exact settings and results are in
[`results/ddm_surrogate_calibration.json`](results/ddm_surrogate_calibration.json),
reproduced by `python3 -m scripts.calibrate_ddm_surrogate`.

Training averages 32 paths per trial. This reduces sampling noise but does not
make finite-sample probability estimates or their gradients exact. Quantized
response times use a straight-through floor gradient; the optional WFPT term
is an uncensored continuous-time auxiliary, not the likelihood of clipped,
lapse-mixed rollout data. Boundary ties and parameters outside the tested grid
remain reasons to check calibration before interpretation.

The model remains a conditional behavioral generator: its recurrent state sees
stimulus strength once per trial and the DDM generates internal noisy evidence.
It is not an observation-stream agent. Comparisons must give competing models
the same information. Signed persistence/exploration heads and a fixed failure
moving average are computational ingredients, not established neural mappings.

## Seed-stability finding

Correcting zero-contrast rewards exposed a Sticky-Q lose-shift spread of 0.50187
across seeds 42, 123, 7, 2024 and 9999 (three 200-trial episodes each). This
variability remains a benchmark limitation. The old arbitrary spread cutoff
was replaced with exact-seed trial replay, schema checks and independent raw
history-count reconciliation. No threshold was widened and no agent behavior
was tuned to pass. Software correctness does not imply seed robustness.

## Validation record

- Final integrated validation: **238 tests passed**, repository-wide Ruff and
  whitespace checks passed, and train/ablation CLI help and report generation
  completed successfully.
- Fresh full-control training smoke: 64 corrected development trials, one epoch,
  followed by 32 PRL trials and four paired 32-trial checkpoint interventions.
  Checkpoint weights remained unchanged during intervention evaluation, all trial
  logs were schema-valid and the evaluators completed. These short runs contain
  no reversal and establish wiring only, not adaptation or predictive quality.
- All six frozen replication source/protocol hashes and its frozen model hash
  were checked against the original plan and remain unchanged.
- Adaptive history readouts now retain their inherited win/loss meaning;
  persistence/exploration have separate named outputs. This also prevents history
  distillation from silently targeting different modules in the adaptive family.
- Generic history metrics in preserved evaluation code condition on recorded
  correctness; PRL correctness denotes optimal-option choice. For reward-driven
  PRL interpretation use its task-specific rewarded/unrewarded metrics, not those
  generic correctness-conditioned probes.

The next scientific gate is a new, provenance-bound comparison against simpler
models with equal inputs on held-out animals. No repair or smoke-test result is
an assertion that the architecture is biologically correct or uniquely needed.
