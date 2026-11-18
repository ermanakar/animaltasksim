# Generative comparison protocol — September 19, 2026

## Question and scope

Can a predictor distinguish genuine outcome/evidence-dependent history from
unobserved changing preferences on the existing development animals? This is an
exploratory, internally frozen experiment, not externally preregistered research.
No fresh cohort is acquired. Earlier frozen interfaces, plans and results remain
unchanged. Reaction time, neural causality and controller necessity are not endpoints.

## Models and information

Six models share categorical current stimulus-by-block controls: task-only,
five-trial rewarded/unrewarded choice history plus two previous-evidence
interactions, continuous-drift approximation, drift plus history, three-state
GLM-HMM, and GLM-HMM plus history. History is a generative response-dependence
model, not a claim that the animal implements reinforcement learning. The two
interactions split rewarded/unrewarded choices at contrast magnitude <= .125.
Unlike earlier nuisance designs, this comparison does not include unsigned
previous-success and previous-contrast terms. Scores are only comparable within
this experiment.

Drift is an AR(1) latent intercept (rho .98, innovation SD .20), discretized on
21 equally spaced states from -3 to 3 with Gaussian probability integrated over
bins. Parameters of that process are fixed, not fitted; this restricted model
is **not PsyTrack**. The initial distribution is uniform and resets at each
session, omission or trial gap. The switching model fits three state-specific
GLMs and transition probabilities by exact observed-data likelihood, with
asymmetric intercept offsets of .5 and 1.5 around a training-fitted single-state
GLM. The iteration budget is 1000; the initial zero-initialized 250-iteration
attempt is retained as interrupted numerical qualification, not interpreted recovery. The winning start is selected solely
by penalized training likelihood. It is an implementation of the GLM-HMM model
family, not a reproduction of Ashwood's original fitting pipeline.

All coefficients have ridge penalty 1; switching transitions have a symmetric
Dirichlet pseudo-count .1. Five fixed animal folds reuse the earlier development
fold assignments. The shared sample excludes the first 100 raw trials of each
session, requires at least 80 committed prefix choices and five consecutive
committed predecessors. All committed warm-up choices inform filtering and
training likelihood; only common eligible later choices are scored. Predictions
are computed before the current response updates the state. Backward smoothing
is permitted for training gradients, never for held-out predictions. No held-out
animal changes any global fitted coefficient or transition matrix.

All methods receive true task block information: this is conditional prediction,
not an autonomous agent discovering the task. Equal-animal NLL is primary; intervals
resample labs then animals conditional on the fitted models, and do not include
uncertainty from refitting the training population. Lab intervals with nine labs
are limited. Reused development animals are not independent confirmation.

## Recovery and power

Generate new responses on all observed development schedules, preserving animal,
lab, session, omission and stimulus/block structure. Draw lab intercept SD .2,
animal SD .4 and session SD .2. Sensory logits use signed square-root contrast
with coefficient 4, and block log-odds coefficient .4. Simulate static preferences,
continuous AR drift, three switching sensitivity/bias states, genuine history,
and mixed drift/history. Synthetic zero-contrast feedback uses a newly drawn
correct side from the task prior; no observed response/outcome is reused.

History coefficients are .05, .15 and .4, with lag decay .6 and a same-strength
weak-error interaction. Report oracle conditional KL, which is an upper-information
comparison, **not** the marginal detectable improvement. The grid brackets weak
effects; it does not establish that the generator matches all real behavior.
There are 40 prespecified seeds per cell (360 runs), full development-sized
synthetic cohorts, two fixed complementary animal folds, and complete refitting.
All attempts, including convergence failures, are retained. Model-family winner
frequencies and a history-detection criterion must both be reported.

The conservative history-detection criterion requires the lower 95% clustered
bootstrap gain to be positive against **both** drift-only and switching-only
alternatives. Null cells must have exact binomial upper 95% detection bound <= .10;
history cells must have lower bound >= .80. All fits must converge. These are
practical qualification thresholds, not guarantees of real-world error control.
Sensitivity to drift discretization, alternative dynamics and independently
implemented published baselines is required even if the numerical gate passes.
A failed cell defers confirmation; it must not be dropped or its threshold changed.

## Mathematics in simple English

A model turns a score into a right-choice probability using
`p(right) = 1 / (1 + exp(-score))`. The score contains the current task evidence,
optionally previous choices/outcomes, and optionally an unseen preference state.

Drift evolves as `next preference = .98 * preference + random change`.
A switching model instead has three states and probabilities of moving between
them. We keep probabilities over possible states rather than guessing one state.
Before a trial we average each state's predicted choice probability using those
state probabilities. After the choice we update our belief and move to the next
trial. This is the forward filter.

A model pays `-log(probability assigned to the observed choice)` each trial.
Smaller average loss means better prediction. Recovery asks whether the winner
corresponds to the rule that actually generated artificial choices. The ability
to predict does not itself establish recovery of the generating mechanism.

## Conditional confirmation plan

Only after qualification: select a specific primary contrast from development,
freeze all code, exclusion rules, model settings, stopping rules and cohort IDs,
and calculate sample size using calibrated recovery at the smallest meaningful
effect and realistic lab/session structure. Register the timestamped protocol
before acquiring any new outcomes. No modification after test access. Existing
60 reserved animals have already been used for the earlier replication and are
not a new confirmation cohort. A separate approval/freeze is needed for any new
scientific endpoint; this document does not authorize accessing it automatically.

If qualification fails, the next study targets identifiability (longer sessions,
more sessions per mouse, or a task manipulation producing different predictions).
PRL remains a potential controlled simulation probe; DMS expansion is deferred.

## Established-method references

[PsyTrack authors' code](https://github.com/nicholas-roy/psytrack) and
[GLM-HMM authors' code](https://github.com/zashwood/glm-hmm) provide independent
benchmarks. Their offline state estimates must not be used as causal held-out
predictions. A matched benchmark needs a prespecified causal prediction adapter;
it cannot be replaced by relabeling the restricted models above.

## Reproduction

Install optional acceleration with `pip install -e '.[dev,research]'`, then run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.compare_generative_history`.
The runner writes plan/source/input hashes before fitting, saves every recovery
cell atomically, and resumes only with unchanged settings, inputs and source.
New files are independent of the frozen September 6 replication implementation.
