# Experiment 005: Change Point Detection

## Overview

**Research Question:** *When does the anomaly start?*

Unlike previous experiments that classify each utterance into anomaly levels, this experiment focuses on **detecting the exact transition point** from normal to anomalous communication patterns. This addresses the critical need in aviation safety: knowing **when** to intervene, not just **what** is happening.

## Architecture

```
Input: Sequence of utterances from a flight case
    ↓
BERT Encoder (frozen/fine-tuned)
    ↓
Utterance Embeddings: [u₁, u₂, ..., uₙ]
    ↓
Sliding Window Dissimilarity Computation
    - Window 1: [u₁..u₅] vs Window 2: [u₆..u₁₀]
    - Metric: Cosine dissimilarity or MMD
    ↓
Dissimilarity Curve: [d₁, d₂, ..., dₘ]
    ↓
Change Point Detection
    - Learnable temporal model OR
    - Peak detection heuristic
    ↓
Output: Predicted change point (utterance index)
```

## Novelty & Contribution

1. **First work** to explicitly model anomaly onset detection in CVR analysis
2. **Distribution shift approach**: Detects changes in communication patterns rather than direct classification
3. **Temporal awareness**: Leverages sequential structure for precise timing
4. **Safety relevance**: Early detection enables proactive intervention

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| MAE (utterances) | Mean absolute error in change point prediction | < 3 utterances |
| MAE (minutes) | Converted to time estimate | < 1.5 minutes |
| Accuracy @ ±1 | Exact utterance match rate | > 30% |
| Accuracy @ ±3 | Within 3 utterances | > 60% |
| Early Detection Rate | % predicted before actual change | > 50% |
| Mean Early Margin | Average time detected early | > 2 utterances |

## Configuration

```yaml
# Key parameters
window_size: 5          # Utterances per comparison window
stride: 1               # Fine-grained sliding
shift_metric: "cosine"  # Dissimilarity metric
smoothing_window: 3     # Smooth dissimilarity curve

# Training
mse_weight: 1.0
early_detection_weight: 0.5  # Reward early detection
smoothness_weight: 0.1
```

## Usage

```bash
# From experiment directory
cd experiments/005_change_point
python run.py

# From project root
python -m experiments.005_change_point.run
```

## Expected Results

| Metric | Expected | Rationale |
|--------|----------|-----------|
| MAE | 2-4 utterances | Challenging due to gradual transition |
| Early Detection | 40-60% | Model tends to be conservative |
| Accuracy @ ±3 | 50-70% | With smoothing and temporal context |

## Comparison with Classification Models

| Aspect | Classification (001-004) | Change Point Detection (005) |
|--------|-------------------------|------------------------------|
| Task | What is the anomaly level? | When does anomaly start? |
| Output | 4-class label | Regression (utterance index) |
| Loss | Cross-entropy | MSE + early detection penalty |
| Evaluation | Accuracy, F1 | MAE, early detection rate |
| Use Case | Post-hoc analysis | Real-time monitoring |

## Future Improvements

1. **Multi-scale windows**: Combine different window sizes
2. **Uncertainty quantification**: Confidence intervals for predictions
3. **Online detection**: Streaming change point detection
4. **Domain adaptation**: Transfer to other safety-critical domains

## References

- Truong et al. (2018) - Selective review of offline change point detection methods
- Aminikhanghahi & Cook (2017) - Survey of methods for time series change point detection
- Noort et al. (2021) - Dataset: CVR transcripts
