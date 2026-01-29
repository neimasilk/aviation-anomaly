# Experiment 006: SMOTE-Augmented Training

## Overview

**Problem:** Extreme class imbalance (14:1 NORMAL:CRITICAL ratio) leads to poor CRITICAL recall (~47% in best model).

**Solution:** Aggressive cost-sensitive learning with Focal Loss and strategic oversampling.

**Target:** CRITICAL recall > 70% (safety requirement)

## Approach

### 1. Focal Loss
Addresses class imbalance by focusing on hard examples:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
- `γ = 2.0`: Down-weights easy examples
- `α_t = [1.0, 1.5, 3.0, 10.0]`: Class weights (CRITICAL gets 10x)

### 2. Weighted Random Sampler
- Doubles dataset size through resampling
- Minority classes sampled more frequently
- Maintains temporal structure

### 3. Cost-Sensitive Learning
Explicit misclassification costs:
```
Cost(CRITICAL miss) = 20x Cost(NORMAL miss)
```

## Configuration

```yaml
# Class weights (for loss function)
class_weights:
  NORMAL: 1.0
  EARLY_WARNING: 1.5
  ELEVATED: 3.0
  CRITICAL: 10.0  # 10x penalty

# Focal Loss
focal_gamma: 2.0

# Oversampling
sampler_multiplier: 2.0  # Double dataset size
```

## Expected Results

| Metric | Baseline (003) | Target (006) |
|--------|---------------|--------------|
| CRITICAL Recall | 47% | > 70% |
| Macro F1 | 0.77 | ~0.75 |
| Accuracy | 86% | ~82% |

**Trade-off:** Lower overall accuracy but better safety (fewer missed CRITICAL cases).

## Usage

```bash
cd experiments/006_smote_augmented
python run.py
```

## Novelty

- First application of Focal Loss for aviation safety NLP
- Explicit safety-cost formulation
- Demonstrates trade-off between accuracy and safety

## References

- Lin et al. (2017): Focal Loss for Dense Object Detection
- Chawla et al. (2002): SMOTE
