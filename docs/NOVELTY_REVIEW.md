# Focused novelty review and next discriminator — September 12, 2026

## Conclusion

The broad ideas of evidence-dependent choice history, slow decision-bias drift,
and changing behavioral strategies already have close precedents. We cannot
claim their discovery. The plausible contribution is a reproducible, source-
verified predictive comparison establishing how much evidence-history signal
survives specific stronger controls on this IBL development cohort, followed by
a genuinely new, prespecified test if the models can be distinguished.

This is a focused review, not an exhaustive systematic search or proof that a
particular incremental analysis has never been published. Search terms included
confidence/history/reinforcement in perceptual decisions, slow drift correction,
PsyTrack, and GLM-HMM, including newer citing work. Abstract-only or preliminary
search leads are not treated as evaluated competing methods.

## Closest established work

| Work | Established contribution relevant here | Consequence for our claim |
|---|---|---|
| [Lak et al., 2020](https://elifesciences.org/articles/49834) | Reward-dependent updating varies with preceding decision difficulty across datasets/species; a confidence-informed learning account is proposed. | Difficulty-dependent history itself is not new. Contrast remains only an evidence proxy in our analysis. |
| [Roy et al., 2021](https://pubmed.ncbi.nlm.nih.gov/33412101/) | PsyTrack estimates changing psychophysical weights over time. | Continuous behavioral drift is an established competing description; our simple causal filters are not a reproduction of PsyTrack. |
| [Ashwood et al., 2022](https://www.nature.com/articles/s41593-021-01007-z) | GLM-HMMs describe persistent, switching behavioral strategies. | Discrete state changes are another serious alternative; simple recent-choice filters cannot rule them out. |
| [Gupta and Brody, 2022](https://arxiv.org/abs/2205.10912) | Slow drift can mimic history updates, and a proposed drift correction can distort several true updating strategies; model-based recovery is examined. | Memory-versus-drift is itself an established question. We need recovery checks and careful claims, not a new label for this problem. |
| [IBL, 2021](https://elifesciences.org/articles/63711) | Standardized mouse behavior and task/history/block analyses across laboratories. | Our source-verified analysis extends an existing behavioral resource. Using new animals alone does not make the underlying idea novel. |

A newer close context is [Findling et al., 2025](https://www.nature.com/articles/s41586-025-09226-1),
which studies inferred priors in the same family of IBL tasks. This reinforces
that learning latent block probabilities is established territory; it does not
supply evidence for our proposed residual history claim.

These sources motivate alternatives; they do not demonstrate that our result
matches, refutes or improves on their methods. In particular, our full-block,
animal-disjoint prediction setup differs from Ashwood's analyzed IBL regime.
A method-level novelty claim would require implementation and matched validation
of the actual published alternatives, not merely our simplified controls.

## Next bounded experiment

Question: does five-trial/evidence history still improve prediction after adding
causally estimated session bias and slow choice-bias summaries?

Estimate a stable session preference from a fixed initial prefix; score only
later trials. Estimate slow residual preference using only preceding choices
relative to a stimulus/block predictor fitted on training animals. Every nested
model uses exactly the same subsequent trials. The prefix and state estimates
must never use future scored choices or fit a full-session test-animal intercept.

These are transparent diagnostic controls, not definitive latent-state models.
Their states are inferred from past behavior and can also absorb genuine memory.
Either a surviving or disappearing history gain therefore remains insufficient
to identify a brain mechanism. Simulated bias-only, drift-only, history-only and
mixed animals will test how often these controls confuse the explanations.

## Gate before a fresh cohort

Do not access a new confirmation cohort until the simulated comparisons can
reliably distinguish the proposed claim from the major alternatives across
reasonable settings and sample sizes. If recovery is ambiguous, improve the
experimental design or compare the published generative models first. An
inconclusive recovery result is a reason to defer confirmation, not to select a
favorable threshold or report a causal mechanism.
