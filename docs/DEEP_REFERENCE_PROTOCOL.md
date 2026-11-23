# Deep development reference protocol — November 23, 2026

## Question and scope

The generative comparison could not separate trial history from slow drift or
state switching, and named more within-animal data as the remedy. The adopted
reference holds about 1.4 sessions per animal. This acquisition collects every
standard session of the **existing development animals** so that later analyses
can fit latent-state models within animals and test history in unbiased sessions.

This document covers acquisition only. It is exploratory, internally frozen and
not externally preregistered. It defines no endpoint and makes no claim.

## What is acquired

- Animals: the 83 development animals in `runs/ibl_source_audit/source_manifest.json`.
  No other animal is listed or downloaded.
- The 60 reserved replication animals are excluded and the runner fails closed
  if one appears. Animals outside both groups stay untouched for a future
  confirmation cohort.
- Protocols: exact versioned `biasedChoiceWorld`, `trainingChoiceWorld` and
  `ephysChoiceWorld`. Each session keeps its family label; families are never
  pooled silently.
- Excluded: optogenetic, RPE, habituation and every other named variant, as in
  the frozen replication rule.
- No outcome-based or QC-based selection at acquisition. Every planned session
  is attempted. Sessions without retrievable trial data are recorded as
  unavailable and are not replaced.

## Stages

1. `inventory`: session metadata for each development animal, by animal name.
   No trial arrays are requested.
2. `plan`: the session list, input hashes and script hash are written and
   hashed before any trial array is requested. The plan is never replaced.
3. `download`: raw ALF trial arrays per session, each with a SHA-256. Resumable.
4. `verify`: rehash every cached file and write one manifest of cached and
   unavailable sessions.

## What this does not change

The adopted reference, the source-reconciled candidate, the frozen replication
and all earlier results are untouched. Outputs go to a new directory,
`runs/ibl_deep_reference_v1`, which is local and not tracked.

## Known limits

- These are previously inspected animals. Anything learned from the deeper data
  is development evidence, not confirmation.
- Training sessions use a different contrast set and no block structure; ephys
  sessions add recording hardware. Each family needs its own conversion and
  eligibility rules, written before analysis.
- Raw arrays are not converted or quality-controlled here. Conversion must
  reuse the verified ALF wheel-choice coding and omission handling.
- Twelve development animals entered the adopted reference only through
  optogenetic or RPE variant sessions (16 of its 120 sessions; 13,265 of 86,648
  trials, 15.3%, from two labs). Eleven of them have no standard session of any
  family and contribute nothing here. Their effect on earlier development
  results is assessed in the [variant sensitivity check](VARIANT_SENSITIVITY.md).

## Acquisition result

Completed November 23, 2026. Of 4,024 planned sessions, **3,950 were cached**
(biased 2,116; training 1,636; ephys 198) from **72 animals**; 74 had no
retrievable trial object on the public server and were not replaced. Connection
failures in the first pass were retried; only missing-object errors remain.
The session list with per-session array hashes is tracked in
[`results/deep_reference_v1_sessions.csv`](results/deep_reference_v1_sessions.csv).
The first look at these data is in [DEEP_EXPLORATION](DEEP_EXPLORATION.md).

## Reproduction

ONE-api is an optional acquisition dependency kept outside the runtime environment.

```bash
python -m scripts.fetch_ibl_deep_reference --stage inventory
python -m scripts.fetch_ibl_deep_reference --stage plan
python -m scripts.fetch_ibl_deep_reference --stage download
python -m scripts.fetch_ibl_deep_reference --stage verify
```
