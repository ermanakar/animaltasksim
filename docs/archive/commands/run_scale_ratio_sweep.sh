#!/bin/bash
# Scale ratio sweep: vary hb_scale vs hd_scale at the injection values that hit WS target.
# Hypothesis: shifting from starting-point bias to drift-rate bias preserves psych slope.
# Fixed: win_t=0.3, lose_t=0.08 (near WS/LS targets at current scale ratio)
set -e

WIN_T=0.3
LOSE_T=0.08
EPISODES=20
ROOT="runs/history_injection_scale_ratio"

# Sweep: (hb_scale, hd_scale) combinations
# Current: hb=2.0, hd=0.3 (mostly starting-point bias, degrades psych)
# Hypothesis: hb=0.5, hd=2.0 (mostly drift-rate bias, preserves psych)
declare -a COMBOS=(
    "0.0 0.5"
    "0.0 1.0"
    "0.0 2.0"
    "0.0 3.0"
    "0.5 0.5"
    "0.5 1.0"
    "0.5 2.0"
    "1.0 0.5"
    "1.0 1.0"
    "1.0 2.0"
    "2.0 0.3"
)

echo "======================================================================"
echo "  SCALE RATIO SWEEP"
echo "  win_t=$WIN_T, lose_t=$LOSE_T"
echo "  ${#COMBOS[@]} combos"
echo "======================================================================"

for COMBO in "${COMBOS[@]}"; do
    read -r HB HD <<< "$COMBO"
    NAME="hb${HB}_hd${HD}"
    NAME="${NAME//./_}"
    DIR="$ROOT/$NAME"

    echo ""
    echo "--- hb_scale=$HB, hd_scale=$HD ---"

    python3 scripts/history_injection_diagnostic.py \
        --run-root "$DIR" \
        --win-tendencies "$WIN_T" \
        --lose-tendencies "$LOSE_T" \
        --history-bias-scale "$HB" \
        --history-drift-scale "$HD" \
        --episodes "$EPISODES"
done

echo ""
echo "======================================================================"
echo "  SUMMARY"
echo "======================================================================"
printf "%10s | %10s | %10s | %10s | %12s | %12s\n" \
    "hb_scale" "hd_scale" "win_stay" "lose_shift" "psych_slope" "chrono_slope"
echo "------------------------------------------------------------------------"

for COMBO in "${COMBOS[@]}"; do
    read -r HB HD <<< "$COMBO"
    NAME="hb${HB}_hd${HD}"
    NAME="${NAME//./_}"
    DIR="$ROOT/$NAME"

    python3 -c "
import json, glob, sys
files = glob.glob('$1/*/metrics.json')
if not files:
    print('  (no results)')
    sys.exit(0)
m = json.load(open(files[0]))['metrics']
ws = m['history']['win_stay']
ls = m['history']['lose_shift']
ps = m['psychometric']['slope']
cs = m['chronometric']['slope_ms_per_unit']
hb, hd = '$2', '$3'
print(f'{hb:>10} | {hd:>10} | {ws:10.3f} | {ls:10.3f} | {ps:12.2f} | {cs:12.1f}')
" "$DIR" "$HB" "$HD" 2>/dev/null || echo "  (failed for hb=$HB hd=$HD)"
done

echo ""
echo "Targets: WS=0.724, LS=0.427, psych=13.2, chrono=-36"
echo "Current: hb=2.0, hd=0.3 gives WS=0.734, psych=9.75, chrono=-27.5"
