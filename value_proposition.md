# Value Proposition — AnimalTaskSim

**Tagline:** A defensible yardstick for “animal‑like” behavior in AI — judged on **fingerprints**, not just reward.

## Elevator pitch
AnimalTaskSim turns canonical rodent/primate tasks into **reproducible software benchmarks** and grades agents against **animal behavioral fingerprints** (psychometric/chronometric curves, history effects, lapses, training dynamics). Any policy that writes our `.ndjson` logs gets a **paper‑grade report** you can show to reviewers, PIs, or leadership.

## Who it’s for
- **Neuroscience & cognitive labs** — validate computational models against standardized tasks and metrics.
- **RL/AI researchers & product teams** — make credible “animal‑like” claims with metrics that survive scrutiny.
- **Educators** — run lab‑grade experiments in class with fast, reproducible outputs.

## Customer pain (what breaks today)
- **Reward rate ≠ behavior match.** High scores hide lapses, biases, history effects, and flexibility failures.
- **Ad‑hoc tasks aren’t trusted.** Every group ships a different setup; results aren’t comparable or reproducible.
- **No standard logs.** You can’t benchmark or audit when every run writes different fields.
- **Heavy infrastructure.** 3D sims and custom rigs slow iteration; reviewers still ask for simple, defensible stats.

## What we provide
- **Task‑faithful environments** mirroring rodent/primate paradigms (trial phases, timings, priors).
- **Unified logging schema** (`.ndjson`) with per‑trial records → auditability and CI‑friendly checks.
- **Evaluation suite** that reproduces canonical figures and outputs **HTML/PDF reports**.
- **Baselines** (Sticky‑Q, Bayesian observer, PPO) to contextualize new agents.
- **CPU‑friendly demos** that run in minutes; agents are **framework‑agnostic**.

## Differentiators
- Not a toy task library — a **benchmark** with **pass/fail criteria** on animal‑style statistics.
- **Agent‑agnostic ingestion:** if it logs to the schema, we score it.
- **Reproducibility by design:** frozen interfaces, seeding discipline, schema validation, CI hooks.
- **Leaderboard‑ready**: reports and metrics can power public or private rankings when you’re ready.

## Outcomes & ROI
- **Labs/PI:** faster model iteration, replication‑grade figures, easier papers & reviews.
- **RL teams:** credible external benchmark → stronger claims, better ablations, cleaner releases.
- **Educators:** plug‑and‑play labs; students reproduce figures and compare agent strategies.

## Why now
Open animal‑behavior datasets and standardized protocols exist; AI needs credible, human/animal‑aligned benchmarks. We productize the yardstick that ties them together.

## Packaging & pricing (indicative)
- **Open Core (MIT):** code, local reports, baselines, tests.
- **Pro Hosted Evaluation:** private uploads, DOI‑citable reports, long‑run storage, SSO, private leaderboards. *Per‑run or seat pricing.*
- **Enterprise:** on‑prem, custom tasks/metrics, SLAs, security reviews.

## Competitive alternatives (and why we win)
- **NeuroGym / PsychRNN:** excellent task libraries, but not a **benchmark** tied to animal data/fingerprints.
- **Animal‑AI Testbed:** broad cognition sandbox, not task‑faithful replication with animal‑style scoring.
- **Ad‑hoc lab code:** incomparable setups, no shared schema, hard to audit.

## Proof points (MVP)
- **Tasks:** Mouse IBL‑style 2AFC, Macaque RDM.  
- **Baselines:** Sticky‑Q, Bayesian observer, PPO.  
- **Reports:** psychometric/chronometric curves, history kernels, bias & lapse parameters, training dynamics.

## KPIs we track
- # of unique agents evaluated • # of reports generated • time‑to‑first‑report (median) • # of labs/classes using it • % of runs passing schema validation

## Roadmap signals
- **v0.1:** IBL 2AFC + RDM, baselines, reports.  
- **v0.2:** **PRL (flexibility)** and **DMS (working memory)** — same scaffold, added metrics.

## Call to action
- **Run the demo:** `make demo_ibl` then open `out/ibl_report.html`.  
- **Integrate your agent:** write `.ndjson` logs → `scripts/evaluate_agent.py` → share the report.  
- **Talk to us:** want hosted evals or a private leaderboard? We’ll set you up.
