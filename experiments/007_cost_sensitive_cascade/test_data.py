"""Test data loading for Experiment 007."""
import sys
from pathlib import Path
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load config
with open(Path(__file__).parent / "config.yaml") as f:
    config = yaml.safe_load(f)

# Load data
df = pd.read_csv(PROJECT_ROOT / "data" / "cvr_labeled.csv")

# Map labels
label_map = {label: idx for idx, label in enumerate(config['data']['labels'])}
df['label'] = df['label'].map(label_map)

print('=== ORIGINAL DISTRIBUTION ===')
print(df['label'].value_counts().sort_index())
print()

# Test new binary strategy
binary_labels = []
for case_id, group in df.groupby('case_id'):
    group = group.sort_values('turn_number')
    labels = group['label'].tolist()
    has_anomaly = any(l > 0 for l in labels)
    binary_labels.append(1 if has_anomaly else 0)

print('=== BINARY DISTRIBUTION (NEW STRATEGY) ===')
print(f'NORMAL (0): {binary_labels.count(0)}')
print(f'ANOMALY (1): {binary_labels.count(1)}')
print(f'Ratio: {binary_labels.count(0)}:{binary_labels.count(1)} ({binary_labels.count(0)/len(binary_labels):.1%} : {binary_labels.count(1)/len(binary_labels):.1%})')

if binary_labels.count(1) > 0:
    print('\n[OK] Binary strategy working correctly!')
else:
    print('\n[ERROR] All samples are NORMAL!')
