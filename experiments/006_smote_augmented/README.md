# Experiment 006: SMOTE-Augmented Training

## Status: 🔧 FIXED & READY TO RUN

**Last Updated:** 2026-02-05

---

## Overview

**Problem:** Extreme class imbalance (14:1 NORMAL:CRITICAL ratio) leads to poor CRITICAL recall (~47% in best model).

**Solution:** Aggressive cost-sensitive learning with Focal Loss and strategic oversampling.

**Target:** CRITICAL recall > 70% (safety requirement)

---

## 🐛 Bug Fix (2026-02-05)

### Issue Identified
The original implementation had a **critical bug** in the dataset class:
- Was taking the **first 20 utterances** from each flight case
- Since CRITICAL phases are at the **end** of flights, model never saw them!
- Label was determined by **majority voting** instead of the last utterance's label

### Fix Applied
1. **Replaced `CVRSequenceDataset`** with `SequentialCVRDataset`
2. **Now uses `create_sequences_from_df()`** from Experiment 002 (proven correct)
3. **Sliding window** properly covers entire flight duration
4. **Label from LAST utterance** in window (represents current state)

### Impact
- Model will now see CRITICAL and ELEVATED phases during training
- Expected significant improvement in CRITICAL recall
- May achieve target of >70% CRITICAL recall

---

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

---

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

# Sliding window (FIXED)
window_size: 10
stride: 5
max_utterances: 20
```

---

## Expected Results

| Metric | Baseline (003) | Target (006) |
|--------|---------------|--------------|
| CRITICAL Recall | 47% | > 70% |
| Macro F1 | 0.77 | ~0.75 |
| Accuracy | 86% | ~82% |

**Trade-off:** Lower overall accuracy but better safety (fewer missed CRITICAL cases).

---

## Usage

```bash
cd experiments/006_smote_augmented
python run.py
```

### Output Files
- `outputs/experiments/006/results.json` - Metrics and configuration
- `models/006/best_model.pt` - Best model checkpoint

---

## Files

| File | Description |
|------|-------------|
| `run.py` | Main training script (FIXED version) |
| `config.yaml` | Experiment configuration |
| `README.md` | This file |

---

## Changes Log

| Date | Change | Status |
|------|--------|--------|
| 2026-01-29 | Initial implementation | ❌ Had sliding window bug |
| 2026-02-05 | Fixed sliding window logic | ✅ Ready to run |

---

## Novelty

- First application of Focal Loss for aviation safety NLP
- Explicit safety-cost formulation
- Demonstrates trade-off between accuracy and safety

---

## References

- Lin et al. (2017): Focal Loss for Dense Object Detection
- Chawla et al. (2002): SMOTE
