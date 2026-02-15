import json
from collections import Counter

pos_stim_actions = Counter()
neg_stim_actions = Counter()

with open("data/ibl/reference.ndjson") as f:
    for line in f:
        trial = json.loads(line)
        if trial['stimulus']['contrast'] > 0:
            pos_stim_actions[trial['action']] += 1
        elif trial['stimulus']['contrast'] < 0:
            neg_stim_actions[trial['action']] += 1

print("Positive stimulus actions:", pos_stim_actions)
print("Negative stimulus actions:", neg_stim_actions)

pos_total = sum(pos_stim_actions.values())
neg_total = sum(neg_stim_actions.values())

print("P(right | stim > 0):", pos_stim_actions[1] / pos_total if pos_total > 0 else 0)
print("P(right | stim < 0):", neg_stim_actions[1] / neg_total if neg_total > 0 else 0)