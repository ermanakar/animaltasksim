# AnimalTaskSim v0.1.0 Release Notes

**Release Date:** October 9, 2025  
**Status:** Proof-of-Concept Complete

---

## Overview

First public release of AnimalTaskSim — a framework for benchmarking AI agents against animal behavioral fingerprints on classic perceptual decision-making tasks.

**TL;DR:** We achieved partial behavioral replication (99% bias match, 100% win-stay match) but revealed fundamental limitations in RL agents' ability to match temporal dynamics. The framework successfully diagnoses where and why AI diverges from biology.

---

## What's Included

### 📦 Core Components

✅ **Two benchmark tasks:**
- Mouse 2AFC (IBL-style visual contrast discrimination)
- Macaque RDM (Random-dot motion from Roitman & Shadlen 2002)

✅ **Three baseline agents:**
- Sticky-GLM (tabular Q-learning with history bias)
- DDM (Drift Diffusion Model - mechanistic baseline)
- PPO (Proximal Policy Optimization via Stable-Baselines3)

✅ **Comprehensive evaluation suite:**
- 11 behavioral metrics (psychometric, chronometric, history effects)
- Schema-validated `.ndjson` logging
- Interactive HTML dashboards with color-coded matches
- Automated metrics computation and visualization

✅ **Reference data:**
- 885 mouse trials (International Brain Laboratory)
- 2611 macaque trials (Roitman & Shadlen 2002)

### 📊 Key Results

**Mouse (Sticky-GLM v21):**
- Bias: 99% match ✅
- Psychometric slope: 140% (too steep but ballpark)
- History effects: 79% match ⚠️

**Macaque (PPO v24):**
- Win-stay: 100% PERFECT match ✅
- RT dynamics: FAILED (flat 210ms vs 400-800ms target) ❌
- Lapses: 37-42% vs ~0% target ❌

**Macaque (DDM v2):**
- RT intercept: 81% match ✅
- RT slope: Correct direction (-139 vs -645 target) ✅
- Psychometric: Too shallow (37% target) ❌

### 📈 Dashboards

Three interactive comparison dashboards included:
- `runs/ibl_final_dashboard.html` (167 KB) - Mouse results
- `runs/rdm_final_dashboard.html` (266 KB) - Macaque PPO results
- `runs/rdm_ddm_dashboard.html` (288 KB) - Macaque DDM results

Each contains:
- Side-by-side psychometric curves
- RT vs. evidence strength plots
- History effect bar charts
- Color-coded metrics tables (green = 90-110% match)

---

## Critical Bug Fix

Discovered and fixed catastrophic action encoding bug in reference data:
- IBL mouse data used numeric actions (0=right, 1=left) instead of strings
- Caused P(right) = 0% across all contrasts (impossible!)
- Led to bogus reference metrics (slope=17.83 vs actual 13.18, bias=4.85 vs actual 0.074)

**Fix:** Added action normalization in `eval/metrics.py` (lines 76-84) to handle both formats.

**Impact:** All calibration work prior to this fix was invalid. Metrics recomputed with corrected reference.

---

## Key Findings

### What We Learned

1. **RL optimizes too well:** Agents find shortcuts (instant decisions + confident guesses) that maximize reward without matching animal constraints.

2. **Temporal dynamics need architecture:** Reward shaping alone can't force realistic RT patterns. DDM's mechanistic accumulation beats PPO's learned policy on temporal metrics.

3. **History effects need memory:** One-step Markov policies only achieve 79% match. Need recurrent connections (LSTM) for deeper dependencies.

4. **Some metrics are easier than others:**
   - Easy: Static biases (99% match)
   - Medium: Choice patterns (79-100% match)
   - Hard: Temporal dynamics (0-22% match)

### What Worked

✅ Framework validates its mission: clearly benchmarks where agents succeed/fail vs. animals  
✅ Reproducible pipeline: `.ndjson` → schema validation → metrics → dashboards  
✅ Multi-task support: same codebase for both mouse and macaque tasks  
✅ CPU-friendly: all training <2 hours on laptop  
✅ Type-safe: Python 3.11 with full type hints, zero Pylance errors  

### What Didn't Work (Agent Limitations)

❌ RT dynamics: PPO makes instant decisions regardless of evidence strength  
❌ Deep history: agents too Markovian, can't maintain multi-trial dependencies  
❌ Lapses: agents guess too much (37-42% vs ~0% animal rate)  
❌ RT shaping: added Gaussian reward bonus at 600ms target, didn't change behavior  

---

## Technical Specifications

**Language:** Python 3.11  
**Dependencies:** gymnasium, numpy, scipy, pandas, torch (CPU), stable-baselines3, matplotlib, pydantic, pytest, tyro

**Key Modules:**
- `envs/` - Gymnasium environments (ibl_2afc.py, rdm_macaque.py)
- `agents/` - Baseline agents (sticky_q.py, ddm_baseline.py, ppo_baseline.py)
- `eval/` - Metrics (metrics.py, report.py, dashboard.py, schema_validator.py)
- `scripts/` - CLI tools (calibration.py, evaluate_agent.py, make_dashboard.py)
- `tests/` - Comprehensive test suite (envs, agents, metrics, schema)

**Performance:**
- Training time: <2 hours per agent on CPU
- Memory: <4 GB RAM
- Reproducible: deterministic seeding via `animaltasksim/seeding.py`

---

## Quick Start

```bash
# Install
git clone https://github.com/ermanakar/animaltasksim.git
cd animaltasksim
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Train mouse agent
python scripts/calibration.py sticky \
  --opts.task ibl \
  --opts.n-episodes 25 \
  --opts.output-dir runs/my_mouse

# Train macaque agent
python scripts/calibration.py ppo \
  --opts.task rdm \
  --opts.total-timesteps 250000 \
  --opts.output-dir runs/my_macaque

# Generate comparison dashboard
python scripts/make_dashboard.py \
  --opts.agent-log runs/my_mouse/trials.ndjson \
  --opts.reference-log data/ibl/reference.ndjson \
  --opts.output runs/my_dashboard.html

# Open dashboard
open runs/my_dashboard.html
```

---

## Documentation

- **FINDINGS.md** - Detailed analysis (15 pages, full results + technical insights)
- **SUMMARY.md** - One-page overview for quick sharing
- **README.md** - API reference and usage guide
- **PRD.md** - Product requirements and design decisions
- **AGENTS.md** - Operating guide for contributors

---

## Future Work (v0.2 Roadmap)

### Planned Enhancements

**1. Hybrid DDM+LSTM Architecture (6-9 weeks)**
- LSTM learns history-dependent drift/bound parameters
- DDM provides mechanistic RT generation
- Multi-objective loss: task reward + behavioral cloning + RT matching
- Expected: 90%+ match on all metrics

**2. Additional Tasks**
- Probabilistic Reversal Learning (PRL)
- Delayed Match-to-Sample (DMS)

**3. Richer Metrics**
- Choice autocorrelation (lag 1-5)
- RT quantiles (10th, 25th, 50th, 75th, 90th)
- Switching rates and run-length distributions
- 2nd-order history effects (win-after-win-stay, etc.)

**4. Cross-Validation**
- 80/20 train/test split on animal sessions
- Report generalization gap
- Ensures agents capture general strategy, not memorize sequences

### Research Directions

- Inverse RL: learn reward function that makes agent match animal trajectories
- Behavioral cloning: directly imitate animal action sequences
- Attention mechanisms: add strategic lapses and distractibility
- Meta-learning: rapid adaptation to new task variants

---

## Known Limitations

1. **RT dynamics:** Current agents can't replicate temporal patterns (flat or weak slopes)
2. **Deep history:** Only 1-step Markov, can't capture multi-trial dependencies robustly
3. **Lapses:** Agents too confident (37-42% lapses vs ~0% animals)
4. **Training time:** While CPU-friendly, some runs take 1-2 hours
5. **Animal variability:** Current comparison to single reference session, not population distribution

---

## Breaking Changes from Prior Work

**None** - This is the first public release (v0.1.0)

Future releases will maintain:
- CLI argument stability (frozen contract)
- `.ndjson` schema compatibility
- Metrics computation consistency

---

## Contributors

**Erman Akar** - Initial development, calibration, analysis

---

## Acknowledgments

- International Brain Laboratory for mouse reference data
- Roitman & Shadlen (2002) for macaque reference data
- Stable-Baselines3 for RL implementations
- Gymnasium for environment framework

---

## Citation

```
AnimalTaskSim v0.1.0: A Benchmark for Evaluating Behavioral Replication in AI Agents
https://github.com/ermanakar/animaltasksim
October 2025
```

---

## License

MIT License - see LICENSE file

---

## Support

- **Issues:** https://github.com/ermanakar/animaltasksim/issues
- **Documentation:** See FINDINGS.md, README.md
- **Contact:** via GitHub issues

---

**Next:** Download and explore the dashboards, read FINDINGS.md for detailed analysis, or jump straight to implementing the hybrid architecture!
