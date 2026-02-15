
import json
from collections import defaultdict

with open("runs/r_ddm_choice_only_v4/trials.ndjson") as f:
    trials = [json.loads(line) for line in f]

# Psychometric and Chronometric Analysis
stim_levels = sorted(list(set(t['stimulus']['contrast'] for t in trials)))
choices_by_stim = defaultdict(list)
rt_by_stim = defaultdict(list)

for trial in trials:
    contrast = trial['stimulus']['contrast']
    choices_by_stim[contrast].append(1 if trial['action'] == 'right' else 0)
    rt_by_stim[contrast].append(trial['rt_ms'])

print("Psychometric Curve:")
for stim in stim_levels:
    p_right = sum(choices_by_stim[stim]) / len(choices_by_stim[stim])
    print(f"  Stimulus: {stim:.4f}, P(right): {p_right:.3f}")

print("\nChronometric Curve:")
for stim in stim_levels:
    mean_rt = sum(rt_by_stim[stim]) / len(rt_by_stim[stim])
    print(f"  Stimulus: {stim:.4f}, Mean RT: {mean_rt:.1f} ms")

# History Effects Analysis
win_stay = []
lose_shift = []

for i in range(1, len(trials)):
    prev_trial = trials[i-1]
    current_trial = trials[i]

    if prev_trial['reward'] > 0: # Previous trial was a win
        if prev_trial['action'] == current_trial['action']:
            win_stay.append(1)
        else:
            win_stay.append(0)
    else: # Previous trial was a loss
        if prev_trial['action'] != current_trial['action']:
            lose_shift.append(1)
        else:
            lose_shift.append(0)

if win_stay:
    p_win_stay = sum(win_stay) / len(win_stay)
else:
    p_win_stay = 0.0

if lose_shift:
    p_lose_shift = sum(lose_shift) / len(lose_shift)
else:
    p_lose_shift = 0.0

print("\nHistory Effects:")
print(f"  Win-Stay Probability: {p_win_stay:.3f}")
print(f"  Lose-Shift Probability: {p_lose_shift:.3f}")
