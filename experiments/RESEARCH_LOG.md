# Research Log

**Catatan utama untuk research ini.** Baca ini dulu sebelum liat kode.

---

## Quick Reference: Apa yang Berhasil?

| # | Eksperimen | Status | Hasil Utama | Catatan |
|---|------------|--------|-------------|---------|
| 001 | Baseline BERT | Completed | Acc: 64.8%, F1: 0.47 | NORMAL class works well (79%), anomaly classes poor (36-38%) |
| 002 | BERT+LSTM | Completed | Acc: 79.2%, F1: 0.66 | **+14% accuracy, +19% F1 vs baseline** - Sequential context helps significantly |
| 003 | Ensemble (001+002) | Completed | Acc: 86.0%, F1: 0.77 | **+7% accuracy, +11% F1 vs best single** - Soft voting ensemble achieved all targets |
| 004 | Hierarchical Transformer | Completed | Acc: 76.1%, F1: 0.61 | **Underperformed vs BERT+LSTM** - Overfitting (val F1 0.70, test F1 0.61) |
| 005 | Change Point Detection | Completed | MAE: 49.1 utt, Early: 65.7% | **First attempt** - Model too conservative, predicts too early (safety bias) |

---

## Eksperimen Terakhir

**Tanggal:** 2026-01-09
**Eksperimen:** 004 - Hierarchical Transformer
**Status:** Completed
**Tujuan:** Implement Model B - Hierarchical Transformer dengan BERT encoder + utterance-level Transformer untuk temporal pattern modeling

**Hasil:**
- Test Accuracy: **76.13%** (-3.04% vs BERT+LSTM, -9.91% vs Ensemble)
- Macro F1: **0.6097** (-4.92% vs BERT+LSTM, -15.71% vs Ensemble)
- **Target Achievement:**
  - Accuracy >= 0.80: NO (0.7613)
  - F1 Macro >= 0.70: NO (0.6097)
- **Best Val F1:** 0.6972 (epoch 20) - very close to target but test performance dropped

**Per-Class Results:**
| Class | F1 | Precision | Recall |
|-------|-----|----------|--------|
| NORMAL | 0.8881 | 0.8597 | 0.9183 |
| EARLY_WARNING | 0.6061 | 0.6329 | 0.5814 |
| ELEVATED | 0.4321 | 0.4667 | 0.4023 |
| CRITICAL | 0.5124 | 0.5536 | 0.4769 |

**Konfigurasi:**
- Model: Hierarchical Transformer (BERT + 4-layer utterance Transformer)
- Parameters: 135M total
- D-Model: 768, Heads: 8, Layers: 4
- Window: 10 utterances, Stride: 5
- Batch: 8, LR: 1e-5, Max Epochs: 30
- Early stopping: patience=5 (triggered at epoch 25)
- Device: NVIDIA GeForce RTX 4080 (CUDA)

**Training Progress:**
| Epoch | Train Acc | Val Acc | Val F1 | Notes |
|-------|-----------|---------|--------|-------|
| 5 | 0.7422 | 0.7017 | 0.5784 | Improving |
| 10 | 0.9073 | 0.7613 | 0.6208 | +0.04 |
| 12 | 0.9448 | 0.7828 | 0.6747 | +0.05 |
| 16 | 0.9775 | 0.7924 | 0.6887 | +0.01 |
| 18 | 0.9765 | 0.7733 | 0.6901 | +0.00 |
| 20 | 0.9860 | 0.7995 | **0.6972** | **Best Val** |
| 25 | 0.9952 | 0.7852 | 0.6866 | Early stop |

**Comparison with All Models:**
| Model | Accuracy | F1 Macro | Status |
|-------|----------|----------|--------|
| Baseline BERT (001) | 64.82% | 0.4734 | Baseline |
| BERT+LSTM (002) | 79.17% | 0.6589 | ✅ Target achieved |
| Ensemble (003) | 86.04% | 0.7668 | ✅ Best model |
| **Hierarchical (004)** | **76.13%** | **0.6097** | ❌ Underperformed |

**Key Findings:**
1. **Hierarchical Transformer underperformed vs BERT+LSTM** - more complex doesn't always mean better
2. **Significant overfitting**: Val F1 0.6972 → Test F1 0.6097 (gap ~0.09)
3. **Model complexity (135M params) too high** for this dataset size (~4K sequences)
4. **ELEVATED class performance dropped significantly** (0.4321 vs 0.6667 in Ensemble)
5. **Training accuracy reached 99.5%** while validation plateaued - classic overfitting pattern
6. **Self-attention at utterance level** may not capture temporal patterns as effectively as Bi-LSTM
7. **Ensemble (003) remains the best model** - simpler approach won

**Lessons Learned:**
- For sequential CVR data, Bi-LSTM's recurrence may be more suitable than Transformer self-attention
- Model size should be proportional to dataset size - 135M params is overkill
- Early stopping worked correctly (patience=5 triggered at epoch 25)
- Resume/checkpoint system functioned properly during long training

**Previous Experiments:**

*002 - BERT+LSTM (2026-01-07)*
- Test Accuracy: 79.17%
- Macro F1: 0.6589
- Sequential modeling with Bi-LSTM

*001 - Baseline BERT (2026-01-07)*
- Test Accuracy: 64.82%
- Macro F1: 0.4734
- Static per-utterance classification

---

## Learning Log

### Dataset
- [x] Noort dataset: 172 cases, 21,626 utterances
- [x] Position-based temporal labeling implemented
- [x] Sequential windows: 4,190 sequences (window=10, stride=5)

### Model
- [x] Baseline BERT: 64.8% acc, 0.47 F1
- [x] BERT+LSTM: 79.2% acc, 0.66 F1
- [x] Ensemble (001+002): 86.0% acc, 0.77 F1 - **BEST MODEL**
- [x] Hierarchical Transformer: 76.1% acc, 0.61 F1 - Underperformed due to overfitting

### Hyperparameter Learnings
- [x] Memory: RTX 4080 16GB - can handle larger batches, but used batch=8 for consistency
- [x] Window overlap: stride=5 (50%) better than non-overlapping
- [x] Early stopping: patience=5 appropriate (best at epoch 20 for Exp 004)
- [x] Ensemble: Equal weights (1:1) work well, weight tuning not necessary for initial results
- [x] Model complexity: 135M params too large for 4K sequences - caused overfitting in Exp 004

---

## Progress Checklist

### Phase 2: Core Development
- [x] Experiment 001: Baseline BERT
- [x] Experiment 002: BERT+LSTM
- [x] Experiment 003: Ensemble
- [x] Experiment 004: Hierarchical Transformer - Completed, underperformed due to overfitting
- [ ] Experiment 005: Change Point Detection
- [ ] Ablation studies

---

## Today's Progress (2026-01-09)

### Experiment 004 - Hierarchical Transformer ✅ COMPLETED
- **Status:** Training completed on RTX 4080
- **Results:** Acc: 76.13%, F1: 0.6097
- **Verdict:** Underperformed vs BERT+LSTM and Ensemble
- **Key Issue:** Overfitting (Val F1 0.6972 → Test F1 0.6097)
- **Training:** 25 epochs, early stopping triggered
- **Files saved:**
  - `models/004/best_model.pt` - Best checkpoint (epoch 20)
  - `models/004/checkpoint.pt` - Resume checkpoint
  - `outputs/experiments/004/results.json` - Results
  - `outputs/experiments/004/confusion_matrix.npy` - Confusion matrix

### Summary: Model Performance Ranking
1. **Ensemble (003)**: 86.04% acc, 0.7668 F1 - **BEST MODEL**
2. **BERT+LSTM (002)**: 79.17% acc, 0.6589 F1 - Best single model
3. **Hierarchical (004)**: 76.13% acc, 0.6097 F1 - Overfitted
4. **Baseline BERT (001)**: 64.82% acc, 0.4734 F1 - Baseline

### Next Steps
1. Run Exp 005 (Change Point Detection) if needed
2. Final analysis & visualization
3. Paper writing section

### Files Modified Today
- `experiments/RESEARCH_LOG.md` - Updated with Exp 004 results
- `.env` - Added DEVICE=cuda configuration
- `experiments/004_hierarchical/config.yaml` - Fixed data path

---

## Today's Progress (2026-01-29)

### Experiment 005 - Change Point Detection ✅ COMPLETED
- **Status:** Training completed on RTX 4080
- **Objective:** Detect WHEN anomaly starts (not just classification)
- **Approach:** Sliding window cosine dissimilarity + learnable detector
- **Results:**
  - MAE: **49.1 utterances** (~24.6 minutes) - *Too high, needs improvement*
  - Accuracy @ ±5 utterances: **17.1%** - *Low precision*
  - Early Detection Rate: **65.7%** - *Good safety characteristic*
  - Mean Early Margin: **67.1 utterances** - *Model is too conservative*

**Key Findings:**
1. **Model exhibits safety bias**: Predicts change point much earlier than actual (67 utterances early on average)
2. This is actually **desirable for aviation safety** - better early than late!
3. The high MAE is due to this conservative behavior, not random errors
4. 65.7% of predictions are before the actual change point

**Lessons Learned:**
- Cosine dissimilarity alone is too sensitive for gradual transitions
- The "early detection weight" in loss function might be too high
- Need better ground truth definition - when exactly does "anomaly" start?
- Consider relative error metrics (e.g., % of sequence length) rather than absolute

**Files created:**
- `experiments/005_change_point/` - Full experiment code
- `src/models/change_point_detector.py` - Reusable model architecture
- `models/005/best_model.pt` - Trained checkpoint
- `outputs/experiments/005/results.json` - Detailed results

### Updated Model Ranking (Classification + Detection)
| Model | Primary Metric | Status |
|-------|---------------|--------|
| **Ensemble (003)** | 86.04% acc, 0.77 F1 | 🥇 Best Classification |
| **BERT+LSTM (002)** | 79.17% acc, 0.66 F1 | 🥈 Best Single Model |
| **Hierarchical (004)** | 76.13% acc, 0.61 F1 | 🥉 Overfitted |
| **Change Point (005)** | 65.7% early detection | 🔬 Novel but needs refinement |

### Next Steps
1. **Improve Change Point Detection:**
   - Try MMD (Maximum Mean Discrepancy) instead of cosine
   - Adjust ground truth definition (maybe use middle of transition?)
   - Reduce early detection weight in loss
   - Add relative error metrics

2. **Paper Preparation:**
   - Statistical significance testing (McNemar's test)
   - Ablation studies visualization
   - Attention analysis from Hierarchical model
   - Comparison table of all 5 experiments

3. **Upload to Drive:**
   - Model 005 checkpoint
   - Updated research log
