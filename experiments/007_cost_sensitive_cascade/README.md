# Experiment 007: Cost-Sensitive Cascade Model

## Overview

**Goal:** Maximize CRITICAL recall to >90% using explicit safety costs

**Approach:** Two-stage cascade architecture
- **Stage 1:** Binary anomaly detector (NORMAL vs ANOMALY) - targets 95% recall
- **Stage 2:** Cost-sensitive 4-class classifier - 20x penalty for CRITICAL misses

## Architecture

```
Input: Sequence of utterances
    ↓
Stage 1: Binary Detector (BERT+LSTM)
    - Classifies: NORMAL vs ANOMALY
    - Target: 95% anomaly recall (accept high false positives)
    - Class weights: [1.0, 5.0]
    ↓
If ANOMALY detected:
    Stage 2: Cost-Sensitive Classifier (BERT+LSTM)
        - Classifies: EARLY_WARNING / ELEVATED / CRITICAL
        - Cost matrix penalizes CRITICAL misses 20x
        - Class weights: [1.0, 2.0, 5.0, 20.0]
Else:
    Output: NORMAL
```

## Key Innovation

**Explicit Cost Matrix:**
```python
Cost Matrix (True → Predicted):
          NORMAL  EARLY  ELEVATED  CRITICAL
NORMAL      1.0    2.0      5.0      10.0
EARLY       2.0    1.0      3.0       8.0
ELEVATED    5.0    3.0      1.0       5.0
CRITICAL   20.0   15.0      5.0       1.0  ← 20x penalty!
```

This reflects real-world safety costs: missing a CRITICAL situation is 20x worse than a false alarm.

## Robust Checkpointing

### Features
- ✅ **Auto-save every epoch** - Never lose progress
- ✅ **Auto-resume** - Run again after interruption, continues automatically
- ✅ **Best model tracking** - Separate best model saved
- ✅ **Progress tracking** - JSON file shows training history
- ✅ **Graceful shutdown** - Ctrl+C saves checkpoint before exit

### Checkpoint Files
```
models/007/
├── checkpoint.pt          # Latest checkpoint (auto-resume)
├── stage1_best.pt         # Best Stage 1 model
├── stage2_best.pt         # Best Stage 2 model (after training)
├── cascade_best.pt        # Best cascade model
└── progress.json          # Training progress tracker
```

### Usage

```bash
# Start training
cd experiments/007_cost_sensitive_cascade
python run.py

# If interrupted (Ctrl+C, power loss, etc.), simply run again:
python run.py
# → Automatically resumes from last checkpoint

# Monitor progress
cat models/007/progress.json
```

## Expected Results

| Metric | Target | Baseline (003) |
|--------|--------|----------------|
| CRITICAL Recall | >90% | 47% |
| Overall Accuracy | ~80% | 86% |
| Macro F1 | ~0.75 | 0.77 |

**Trade-off:** Lower overall accuracy but much better safety (fewer missed critical situations).

## Configuration

See `config.yaml` for:
- Cost matrix values
- Class weights
- Training hyperparameters
- Checkpoint settings

## References

- Elkan, C. (2001). The foundations of cost-sensitive learning
- Lin et al. (2017). Focal Loss for Dense Object Detection
- Bach et al. (2024). Cost-Sensitive Learning for Safety-Critical NLP
