#!/bin/bash
# Co-evolution v4: final fine-tuning around the sweet spot.
#
# v3 showed win=0.30/lose=0.15 gives psych=13.16, chrono=-34.6 (on target)
# but WS=0.704 (need 0.724) and LS=0.449 (need 0.427).
# Interpolation suggests win≈0.31, lose≈0.17.
set -e

COMMON_ARGS=(
    --task=ibl_2afc
    --drift-scale=10.0
    --drift-magnitude-target=9.0
    --lapse-rate=0.05
    --episodes=20
    --history-bias-scale=2.0
    --history-drift-scale=0.3
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

WIN_TS=(0.30 0.31 0.32)
LOSE_TS=(0.15 0.16 0.17 0.18)

TOTAL=$(( ${#WIN_TS[@]} * ${#LOSE_TS[@]} ))
COUNT=0

for WIN_T in "${WIN_TS[@]}"; do
    for LOSE_T in "${LOSE_TS[@]}"; do
        COUNT=$((COUNT + 1))
        NAME="win${WIN_T}_lose${LOSE_T}"
        NAME="${NAME//./_}"

        echo ""
        echo "======================================================================"
        echo "  [$COUNT/$TOTAL] win_t=$WIN_T, lose_t=$LOSE_T (target=9)"
        echo "======================================================================"
        python3 scripts/train_hybrid_curriculum.py \
            --output-dir="runs/coevolution_v4/${NAME}_seed42" \
            --seed=42 \
            --inject-win-tendency="$WIN_T" \
            --inject-lose-tendency="$LOSE_T" \
            "${COMMON_ARGS[@]}"
        python3 scripts/evaluate_agent.py --run "runs/coevolution_v4/${NAME}_seed42"
    done
done

echo ""
echo "======================================================================"
echo "  SWEEP COMPLETE"
echo "======================================================================"

for WIN_T in "${WIN_TS[@]}"; do
    for LOSE_T in "${LOSE_TS[@]}"; do
        NAME="win${WIN_T}_lose${LOSE_T}"
        NAME="${NAME//./_}"
        DIR="runs/coevolution_v4/${NAME}_seed42"
        python3 -c "
import json, sys
m = json.load(open(sys.argv[1]))['metrics']
ps = m['psychometric']['slope']
cs = m['chronometric']['slope_ms_per_unit']
ws = m['history']['win_stay']
ls = m['history']['lose_shift']
lp = m['psychometric']['lapse_low']
print(f'win={sys.argv[2]:>4} lose={sys.argv[3]:>4} | psych={ps:5.2f} | chrono={cs:6.1f} | WS={ws:.3f} | LS={ls:.3f} | lapse={lp:.3f}')
" "$DIR/metrics.json" "$WIN_T" "$LOSE_T" 2>/dev/null || echo "  (failed: win=$WIN_T lose=$LOSE_T)"
    done
done

echo ""
echo "Targets: psych=13.2, chrono=-36, WS=0.724, LS=0.427, lapse=0.05"
