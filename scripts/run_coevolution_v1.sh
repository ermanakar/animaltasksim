#!/bin/bash
# Co-evolution training: fixed history injection + learning evidence circuits.
#
# The injection diagnostic proved win_t=0.3/lose_t=0.08 produces WS≈0.724,
# but degrades psych slope to ~10 (target 13.2) because the pretrained
# evidence circuits never learned to compensate for history bias.
#
# This experiment trains from scratch with injection active, so the LSTM
# and DDM heads co-evolve alongside history effects — they learn to be
# more evidence-sensitive to overcome the bias on contradicting trials.
set -e

COMMON_ARGS=(
    --task=ibl_2afc
    --drift-scale=10.0
    --drift-magnitude-target=6.0
    --lapse-rate=0.05
    --episodes=20
    --history-bias-scale=2.0
    --history-drift-scale=0.3
    --inject-win-tendency=0.3
    --inject-lose-tendency=0.08
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

for SEED in 42 123 456; do
    echo ""
    echo "======================================================================"
    echo "  SEED $SEED — co-evolution with inject_win=0.3, inject_lose=0.08"
    echo "======================================================================"
    python3 scripts/train_hybrid_curriculum.py \
        --output-dir="runs/coevolution_v1/seed${SEED}" \
        --seed="$SEED" \
        "${COMMON_ARGS[@]}"
    python3 scripts/evaluate_agent.py --run "runs/coevolution_v1/seed${SEED}"
done

echo ""
echo "======================================================================"
echo "  ALL SEEDS COMPLETE — Results:"
echo "======================================================================"
for SEED in 42 123 456; do
    echo ""
    echo "--- seed${SEED} ---"
    python3 -c "
import json
m = json.load(open('runs/coevolution_v1/seed${SEED}/metrics.json'))['metrics']
print(f\"  psych_slope:  {m['psychometric']['slope']:.2f}\")
print(f\"  chrono_slope: {m['chronometric']['slope_ms_per_unit']:.1f}\")
print(f\"  win_stay:     {m['history']['win_stay']:.3f}\")
print(f\"  lose_shift:   {m['history']['lose_shift']:.3f}\")
print(f\"  lapse:        {m['psychometric']['lapse_low']:.3f} / {m['psychometric']['lapse_high']:.3f}\")
print(f\"  bias:         {m['psychometric']['bias']:.4f}\")
"
done

echo ""
echo "Targets: psych=13.2, chrono=-36, WS=0.724, LS=0.427"
