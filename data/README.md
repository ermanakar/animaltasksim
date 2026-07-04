# Reference data — provenance & licensing

> **Licensing note:** Code in this repository is **MIT** (see `/LICENSE`). The
> reference **data** in this directory are **not** MIT — they are derived from
> third-party datasets under their own licenses, described below. Do not treat
> the data as MIT-licensed.

## IBL mouse 2AFC (`data/ibl/`)

`data/ibl/reference.ndjson` (and the preserved `reference_10session.ndjson`,
`reference_single_session.ndjson`) contain a **derived subset** of publicly
released **International Brain Laboratory (IBL)** behavioral data, accessed from
the IBL public server **OpenAlyx** (`https://openalyx.internationalbrainlab.org`)
via the `ONE-api` client in **July 2026** using `scripts/fetch_ibl_reference.py`.

**What is included (derived / transformed):** only selected trial-level
behavioral fields — signed stimulus contrast, choice (left/right), correctness,
reward, reaction time (computed as `response_times − stimulus_onset`), and the
block prior — converted to this project's NDJSON schema, plus a manifest of
session identifiers (EIDs) at `data/ibl/reference.manifest.json`. No
electrophysiology, video, or raw traces are included.

**License:** IBL publicly released data are distributed under **Creative Commons
Attribution 4.0 International (CC-BY 4.0)**
(<https://creativecommons.org/licenses/by/4.0/>). This derived dataset is
redistributed under the same **CC-BY 4.0** license, with attribution below.

**Attribution / changes:** Original creator — the International Brain Laboratory.
Changes made — data were filtered to trained, QC-passing `biasedChoiceWorld`
sessions and the biased-blocks contrast set, transformed to NDJSON, and reaction
time was computed as `response_times − stimulus_onset`.

**Please cite** when using this reference:

- The International Brain Laboratory et al. "Standardized and reproducible
  measurement of decision-making in mice." *eLife* 10:e63711, 2021.
  DOI: <https://doi.org/10.7554/eLife.63711>
- IBL Behavioral Data on AWS Open Data — accessed July 2026 from
  <https://registry.opendata.aws/ibl-behaviour/>
- (Data architecture) The International Brain Laboratory et al. "A modular
  architecture for organizing, processing and sharing neurophysiology data."
  *Nature Methods*, 2023. DOI: <https://doi.org/10.1038/s41592-022-01742-6>

## Macaque RDM (`data/macaque/`)

`data/macaque/reference.ndjson` is derived from the classic random-dot-motion
dataset of **Roitman & Shadlen (2002)**, *Journal of Neuroscience* 22(21):9475–9489.
Cite the original publication when using it.
