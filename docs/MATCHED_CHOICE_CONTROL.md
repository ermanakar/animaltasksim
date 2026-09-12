# Matched choice-only architecture comparison — September 12, 2026

This completes the missing reduced-model control from the earlier fitting diagnosis.
The same 12 fitting and eight development evaluation mice, seeds 42 and 123,
hidden size 16, 64-trial training chunks, optimizer, six-epoch schedule and
512-path prediction procedure are retained. Both architectures have RT loss weight
zero. The saved full-model predictions are reused unchanged; only the reduced
model is newly fitted. The primary endpoint is epoch six, not the best checkpoint.

## Result

| Model | Epoch | Equal-animal choice NLL | RT MSE (seconds squared) |
|---|---:|---:|---:|
| choice_only_full_control | 2 | 0.467737 | 11.942554 |
| choice_only_full_control | 4 | 0.455407 | 11.970792 |
| choice_only_full_control | 6 | 0.446538 | 12.035338 |
| choice_only_no_control | 2 | 0.460151 | 11.909983 |
| choice_only_no_control | 4 | 0.456669 | 11.918394 |
| choice_only_no_control | 6 | 0.445847 | 12.010433 |
| simple_history | 0 | 0.464504 | 11.245851 |

Seeds are averaged within animals; eight contribute choice scores and seven
contribute RT scores. RT has no direct training objective here and is diagnostic.

Positive improvement below favors the reduced model.

| Comparator at primary endpoint | Reduced-model choice gain | Descriptive animal-bootstrap 95% interval | Mice favoring reduced |
|---|---:|---|---:|
| Choice-only full model | 0.000690 | [-0.004550, 0.007200] | 2/8 |
| Simple history | 0.018656 | [-0.001192, 0.040604] | 5/8 |

## Interpretation

The reduced model has slightly lower mean choice NLL (0.445847 versus 0.446538),
but its advantage is only 0.000690 nats/trial, with an interval spanning
−0.004550 to 0.007200. Only two of eight animals favor reduced; six favor full,
so the aggregate difference also hides heterogeneity. This is inconclusive
about predictive superiority and does not demonstrate equivalence. The adaptive
controller's necessity remains unestablished under the fixed fitting budget.

Reduced also has lower mean NLL than simple history, but that comparison's
interval spans zero (−0.001192 to 0.040604). Do not infer a confirmed baseline
victory from the mean alone. Mean RT MSE remains worse than simple history and
both recurrent models have no direct timing supervision in this comparison.

Retain reduced as a parsimonious comparison model and full as a hypothesis;
these data do not justify expanding or deleting the adaptive architecture.

## Verification and limits

Independent checks recomputed every source-linked prediction loss, all animal
means and comparison bootstrap intervals. The saved full predictions match the
earlier run. Plan, runner, dependency and input hashes are unchanged. The full
repository suite passed 257 tests; Ruff passed.

Earlier incomplete runner attempts stopped during import/serialization and
matched-score/provenance corrections. They provide no reported result. The final
run uses a new directory and a new frozen plan; its complete schedule is retained.

This uses already inspected development animals and only two training seeds.
Equal epochs do not imply equal convergence. Intervals condition on these fits
and omit training-population and Monte Carlo uncertainty; comparisons are
exploratory and not multiplicity-adjusted. Neither architecture is validated as
a neural mechanism or a joint choice/RT explanation. A predictive advantage
under these settings does not establish biological necessity or universal
architectural superiority.

Absolute scores cannot be compared with the competing-history study, which
uses a different cohort, common-history eligibility and analyst block controls.

[Plan, results and verification](results/matched_choice_control_v1.json) ·
[Per-animal scores](results/matched_choice_control_v1_subject_scores.csv) ·
[Training curves](results/matched_choice_control_v1_training_curves.csv)

## Reproduction

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m scripts.choice_control_comparison`
with a fresh output directory. The runner requires the saved matched study and
refuses to overwrite a nonempty destination.
