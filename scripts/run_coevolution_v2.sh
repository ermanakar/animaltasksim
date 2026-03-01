#!/bin/bash
# Co-evolution v2: sweep drift_magnitude_target to recover psych slope.
#
# v1 showed WS=0.714 (target 0.724) and chrono=-33 (target -36) but
# psych dropped to 10.1 (target 13.2). The evidence circuits need
# higher drift_gain to compensate for history bias on easy trials.
#
# Also bumps lose_t from 0.08 to 0.12 to push LS closer to 0.427.
#
# Sweep: drift_magnitude_target × {7, 8, 9, 10} at seed=42 (discovery phase)
set -e

COMMON_ARGS=(
    --task=ibl_2afc
    --drift-scale=10.0
    --lapse-rate=0.05
    --episodes=20
    --history-bias-scale=2.0
    --history-drift-scale=0.3
    --inject-win-tendency=0.3
    --inject-lose-tendency=0.12
    --no-use-default-curriculum
    --no-allow-early-stopping
    --phase1-epochs=15
    --phase1-choice-weight=0.0
    --phase1-rt-weight=1.0
    --phase1-history-weight=0.0
    --phase1-drift-magnitude-weight=0.5
    --phase2-epochs=10
    --phase2-choice-weight=0.5
    --phase2-rt-weight=0.8
    --phase2-history-weight=0.1
    --phase2-drift-magnitude-weight=0.5
    --phase3-epochs=10
    --phase3-choice-weight=1.0
    --phase3-rt-weight=0.5
    --phase3-history-weight=0.2
    --phase3-drift-magnitude-weight=0.5
)

for TARGET in 7 8 9 10; do
    echo ""
    echo "======================================================================"
    echo "  drift_magnitude_target=$TARGET — inject win=0.3, lose=0.12"
    echo "======================================================================"
    python3 scripts/train_hybrid_curriculum.py \
        --output-dir="runs/coevolution_v2/target${TARGET}_seed42" \
        --seed=42 \
        --drift-magnitude-target="$TARGET" \
        "${COMMON_ARGS[@]}"
    python3 scripts/evaluate_agent.py --run "runs/coevolution_v2/target${TARGET}_seed42"
done

echo ""
echo "======================================================================"
echo "  SWEEP COMPLETE — Results:"
echo "======================================================================"
printf "%8s | %12s | %12s | %10s | %10s | %10s\n" \
    "target" "psych_slope" "chrono_slope" "win_stay" "lose_shift" "lapse_low"
echo "------------------------------------------------------------------------"

for TARGET in 7 8 9 10; do
    python3 -c "
import json
m = json.load(open('runs/coevolution_v2/target${TARGET}_seed42/metrics.json'))['metrics']
ps = m['psychometric']['slope']
cs = m['chronometric']['slope_ms_per_unit']
ws = m['history']['win_stay']
ls = m['history']['lose_shift']
lp = m['psychometric']['lapse_low']
print(f'${TARGET:>8} | {ps:12.2f} | {cs:12.1f} | {ws:10.3f} | {ls:10.3f} | {lp:10.3f}')
" 2>/dev/null
done

echo ""
echo "Targets: psych=13.2, chrono=-36, WS=0.724, LS=0.427, lapse=0.05"
echo "v1 baseline (target=6): psych=10.1, chrono=-33, WS=0.714, LS=0.488"
