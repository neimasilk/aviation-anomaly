# Experiment 003: Ensemble Baseline + Sequential

**Status:** In Progress
**Created:** 2026-01-08
**Tags:** ensemble, voting, baseline, sequential

## Overview

Soft voting ensemble combining Experiment 001 (Baseline BERT) and Experiment 002 (BERT+LSTM) for improved anomaly detection. The ensemble uses weighted probability averaging with weight tuning on the validation set.

## Hypothesis

Combining the per-utterance baseline classifier with the sequential BERT+LSTM model will improve performance by:
1. Leveraging complementary strengths of both approaches
2. Reducing variance through model diversity
3. Improving confidence on borderline cases

**Target:** Accuracy >= 0.80, Macro F1 >= 0.70 (+5-7% improvement over best single model)

## Base Models

| Model | Experiment | Checkpoint | Test Acc | Macro F1 |
|-------|-----------|------------|----------|----------|
| Baseline BERT | 001 | `models/001/best_model.pt` | 0.6482 | 0.4734 |
| BERT+LSTM | 002 | `models/002/best_model.pt` | 0.7917 | 0.6589 |

## Ensemble Configuration

- **Type:** Soft Voting (probability averaging)
- **Combination Method:** Weighted Average
- **Weight Tuning:** Grid search on validation set
  - Range: [0.0, 2.0]
  - Steps: 0.1

## Usage

```bash
# Run from project root
python experiments/003_ensemble/run.py

# Or with specific device
python experiments/003_ensemble/run.py --device cuda
```

## Output

Results will be saved to:
- `outputs/experiments/003/` - Metrics, comparisons, confusion matrices
- `logs/003/` - Training/inference logs

## Expected Results

| Metric | Exp 001 | Exp 002 | Exp 003 (Target) |
|--------|---------|---------|------------------|
| Accuracy | 0.6482 | 0.7917 | >= 0.80 |
| Macro F1 | 0.4734 | 0.6589 | >= 0.70 |

## Files

- `config.yaml` - Experiment configuration
- `run.py` - Ensemble implementation and evaluation script
- `README.md` - This file

## References

- Experiment 001: `experiments/001_baseline/`
- Experiment 002: `experiments/002_bert_lstm/`
