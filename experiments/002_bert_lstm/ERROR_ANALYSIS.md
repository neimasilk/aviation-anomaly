# Error Analysis: Experiment 002 BERT+LSTM

## Confusion Matrix Comparison

### Baseline (001) - Static BERT
| Actual \ Predicted | NORMAL | EARLY_WARNING | ELEVATED | CRITICAL |
|-------------------|--------|---------------|----------|----------|
| NORMAL | 1706 | 321 | 65 | 12 |
| EARLY_WARNING | 348 | 230 | 58 | 10 |
| ELEVATED | 138 | 59 | 109 | 17 |
| CRITICAL | 48 | 27 | 31 | 44 |

### BERT+LSTM (002) - Sequential
| Actual \ Predicted | NORMAL | EARLY_WARNING | ELEVATED | CRITICAL |
|-------------------|--------|---------------|----------|----------|
| NORMAL | 357 | 23 | 6 | 0 |
| EARLY_WARNING | 33 | 86 | 9 | 1 |
| ELEVATED | 9 | 16 | 32 | 8 |
| CRITICAL | 4 | 7 | 15 | 23 |

## Metrics Comparison

| Class | Recall (001) | Recall (002) | Improvement | Precision (001) | Precision (002) |
|-------|-------------|-------------|-------------|----------------|----------------|
| NORMAL | 81.1% | 92.5% | +11.4% | 76.2% | 88.6% |
| EARLY_WARNING | 35.6% | 66.7% | +31.1% | 36.1% | 65.2% |
| ELEVATED | 33.7% | 49.2% | +15.5% | 41.4% | 51.6% |
| CRITICAL | 29.3% | 46.9% | +17.6% | 53.0% | 71.9% |

## Critical Safety Analysis

### Missed Detections (Anomaly -> NORMAL) - SAFETY ISSUE
| Class | Baseline | Exp 002 | Reduction |
|-------|----------|---------|-----------|
| EARLY_WARNING | 348 (53.9%) | 33 (25.6%) | 91% |
| ELEVATED | 138 (42.7%) | 9 (13.8%) | 93% |
| CRITICAL | 48 (32.0%) | 4 (8.2%) | 92% |
| **TOTAL** | **534** | **46** | **91%** |

### Severity Underestimation (Predicted as LESS severe)
| Pattern | Baseline | Exp 002 |
|---------|----------|---------|
| EARLY_WARNING -> NORMAL | 348 | 33 |
| ELEVATED -> NORMAL/EARLY | 197 | 25 |
| CRITICAL -> Less Severe | 106 | 26 |
| **TOTAL DANGEROUS PREDICTIONS** | **651** | **84** |

## Key Findings

1. **Sequential modeling reduces missed detections by 91%**
   - From 534 to 46 anomalies misclassified as NORMAL
   - This is critical for aviation safety

2. **EARLY_WARNING detection improved most (+31.1% recall)**
   - From 35.6% to 66.7%
   - Important for early intervention

3. **Model is conservative for CRITICAL predictions**
   - High precision (71.9%): when predicted, likely correct
   - Lower recall (46.9%): trade-off for reliability

4. **Remaining challenges**
   - ELEVATED has lowest recall (49.2%)
   - Adjacent level confusion (EARLY_WARNING <-> ELEVATED)
   - CRITICAL recall still below 50%

## Error Patterns in Exp 002

### ELEVATED (hardest class)
- 32/65 correct (49.2%)
- 16 confused as EARLY_WARNING (48.5% of errors)
- 8 confused as CRITICAL (24.2% of errors)
- 9 confused as NORMAL (27.3% of errors)

### CRITICAL (conservative)
- 23/49 detected (46.9%)
- 15/26 errors -> ELEVATED (57.7% - underestimation)
- 7/26 errors -> EARLY_WARNING (26.9%)
- 4/26 errors -> NORMAL (15.4%)

### EARLY_WARNING (much improved)
- 86/129 detected (66.7%)
- 33/129 as NORMAL (25.6%)
- 9/129 as ELEVATED (7.0%)
- Only 1 as CRITICAL (0.8%)

## Recommendations

### For Next Experiments

1. **Hierarchical Transformer (Exp 003)**
   - May better capture long-range dependencies
   - Could improve ELEVATED/CRITICAL boundary

2. **Threshold Optimization**
   - Lower CRITICAL threshold for higher recall
   - Accept more false alarms for safety

3. **Ensemble Methods**
   - Combine baseline + sequential predictions
   - May improve boundary cases

4. **Focal Loss**
   - Address class imbalance better than weighted loss
   - Focus on hard-to-classify examples

### Dataset Considerations

1. **Current dataset is sufficient** - 4,190 sequences
2. **Consider temporal augmentation**:
   - Variable window sizes (5, 10, 15 utterances)
   - Multiple stride rates
3. **Synthetic augmentation for CRITICAL**:
   - Oversample CRITICAL sequences
   - SMOTE-like techniques for sequential data
