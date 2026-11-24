# Deep reference conversion protocol and result — November 23, 2026

Milestone 1 of the [research plan](RESEARCH_PLAN.md): turn the raw arrays of the
[deep development reference](DEEP_REFERENCE_PROTOCOL.md) into schema-valid
trials. Everything above the "Result" section was written before any session was
converted; the result was added afterwards without changing the rules.

**Result: the gate passed.** 3,946 sessions converted; all 104 shared sessions
match the reconciled reference on all 73,383 trials.

## What was already seen

The rules below were written after two looks at these data, and both are stated
so the reader can judge them:

- The [exploratory first look](DEEP_EXPLORATION.md) examined behaviour in the
  raw arrays. No rule here uses a behavioural result from it.
- A structural survey of the 3,950 cached sessions counted missing arrays,
  length mismatches, coding violations, contrast values and prior values per
  family. It found 4 malformed sessions, no coding violations in the rest, 222
  sessions with response times missing for most trials, and 43 sessions under
  100 trials.

## Conversion rules

One session converts as a whole or not at all. Trials are never dropped, so
original indices and adjacency are preserved.

A session is **converted** only if all of these hold:

1. `choice`, `feedbackType`, `contrastLeft`, `contrastRight`,
   `probabilityLeft`, `stimOn_times` and `response_times` are present, equal in
   length and non-empty.
2. Choices are in {−1, 0, +1} and feedback in {−1, +1}.
3. Exactly one of the two contrasts is finite on every trial, and every prior is
   finite.
4. No omitted trial is marked rewarded.
5. On every committed trial with non-zero contrast, the wheel-choice sign agrees
   with feedback (ALF −1 selects the right stimulus, +1 the left).
6. Absolute contrasts lie in the family's protocol set: {0, 0.0625, 0.125,
   0.25, 1} for biased and ephys sessions, with 0.5 added for training sessions.

A session failing any rule is recorded as not converted with the first failing
rule. It is not repaired and not replaced.

Field mapping, identical to the source-reconciled reference:

| Schema field | Source |
|---|---|
| `stimulus.contrast` | right contrast if finite, else minus left contrast |
| `block_prior.p_right` | 1 − `probabilityLeft` |
| `action` | 0 right, 1 left, 2 omission |
| `correct`, `reward` | `feedbackType == 1` |
| `rt_ms` | 1000 × (`response_times` − `stimOn_times`) if finite, positive and committed; else null |
| `prev` | action, reward and correct of the preceding trial; null on the first |

There is no fallback for missing response times. Other fields follow the
reference: `task` "IBL2AFC", `seed` 0, empty `phase_times`. The agent version
names the family.

## No quality filtering

Conversion applies no rule based on performance, trial count, omissions,
response times, history or any inferred state. Short sessions and sessions with
missing response times are converted.

The manifest records descriptive counts for every session: trials, omissions,
response-time coverage, full-contrast trials and full-contrast accuracy
counting omissions as failures. It also records one flag, `meets_frozen_gate`,
computed with the eligibility thresholds frozen for the
[September replication](REPLICATION_PROTOCOL.md) (at least 150 trials, at least
20 full-contrast trials, full-contrast accuracy ≥ 0.85). The flag selects
nothing here. Any later analysis must state in its own protocol whether it
uses it.

## Outputs

`runs/ibl_deep_converted_v1`, local and untracked:

- `<family>/<subject>.ndjson`: one file per animal per family. Families are
  never mixed in a file.
- `manifest.json`: every planned session with its status, identity, session
  order within the animal, descriptive counts, source hash and output hash,
  plus input and code hashes. The schema forbids extra trial fields, so animal,
  lab and family live here.

The runner refuses a non-empty output directory.

## Gate

The 104 standard biased sessions of the adopted reference are also in the deep
reference, downloaded again two months later. For each of them, every converted
trial must equal the corresponding trial of the source-reconciled candidate
(`runs/ibl_source_reconciled/reference.ndjson`) in every field except the agent
version, with response times equal to within 10⁻⁶ ms.

**Pass:** all 104 sessions are converted and every trial matches. Any missing
session or mismatched trial fails the gate, is reported with its location, and
stops milestone 2 until explained. The gate result is written once and never
replaced.

The gate checks the converter against an independently verified conversion of
the same task family. It does not validate training or ephys conversions beyond
the shared rules; those rest on the unit tests and rules 1–6.

## Known limits

- Training-session priors are recorded as given. In 177,645 of 898,639 surveyed
  training trials `probabilityLeft` is not 0.5, taking values from 0 to 1 in
  steps of 0.1. These are not block priors, and analyses that require standard
  priors must reject or handle them explicitly.
- Repeated trials after errors in training sessions are not marked. A `repNum`
  array exists for only 688 sessions and the schema has no field for it.
- Response time follows the reference convention (stimulus onset to response).
  Sessions lacking stimulus-onset times get null response times.
- Converting a session says nothing about whether the animal was trained,
  engaged or healthy in it.

## Result

[Portable summary](results/deep_conversion_v1.json) ·
[Per-session table](results/deep_conversion_v1_sessions.csv)

**Conversion.** 3,946 of 3,950 cached sessions converted, from 72 animals. The
four that did not are the malformed sessions the survey found (three with
missing arrays, one with mismatched lengths). No session failed a coding rule.

| Family | Sessions | Trials | Omissions | Meets frozen gate | Response time for under half of choices | Under 150 trials |
|---|---:|---:|---:|---:|---:|---:|
| Biased | 2,115 | 1,528,423 | 6,097 | 1,839 | 140 | 23 |
| Training | 1,635 | 898,639 | 5,096 | 579 | 84 | 83 |
| Ephys | 196 | 110,875 | 1,698 | 143 | 5 | 6 |

The last three columns are descriptive. Nothing was filtered on them.

**Gate: passed.** All 104 shared standard biased sessions were converted, and
all 73,383 trials equal the source-reconciled reference in every compared
field. The deep arrays were downloaded on November 23 and the reconciled
reference was built from a September download, so the match also shows the
public source did not change for these sessions in between.

**What this does and does not establish.** The biased-session converter
reproduces an independently verified conversion. Training and ephys sessions
pass the same structural rules and unit tests but have no independent reference
to match. No claim is made about data quality or behaviour.

Milestone 2 of the research plan (recovery at the new data depth) may proceed.

## Reproduction

```bash
python -m scripts.convert_ibl_deep_reference --stage convert
python -m scripts.convert_ibl_deep_reference --stage gate
```
