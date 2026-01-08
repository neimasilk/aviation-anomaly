# Research Log

**Catatan utama untuk research ini.** Baca ini dulu sebelum liat kode.

---

## Quick Reference: Apa yang Berhasil?

| # | Eksperimen | Status | Hasil Utama | Catatan |
|---|------------|--------|-------------|---------|
| 001 | Baseline BERT | Completed | Acc: 64.8%, F1: 0.47 | NORMAL class works well (79%), anomaly classes poor (36-38%) |
| 002 | BERT+LSTM | Completed | Acc: 79.2%, F1: 0.66 | **+14% accuracy, +19% F1 vs baseline** - Sequential context helps significantly |
| 003 | Ensemble (001+002) | Completed | Acc: 86.0%, F1: 0.77 | **+7% accuracy, +11% F1 vs best single** - Soft voting ensemble achieved all targets |
| 004 | Hierarchical Transformer | Ready to run | N/A | Model implemented, will run on RTX 4080 at office |

---

## Eksperimen Terakhir

**Tanggal:** 2026-01-08
**Eksperimen:** 003 - Ensemble Baseline + Sequential
**Status:** Completed
**Tujuan:** Combine Baseline BERT (001) and BERT+LSTM (002) via soft voting for improved anomaly detection

**Hasil:**
- Test Accuracy: **86.04%** (+6.87% vs BERT+LSTM, +21.22% vs baseline)
- Macro F1: **0.7668** (+10.79% vs BERT+LSTM, +29.34% vs baseline)
- **Target Achievement:**
  - Accuracy >= 0.80: YES (0.8604)
  - F1 Macro >= 0.70: YES (0.7668)

**Per-Class Results:**
| Class | F1 | Precision | Recall |
|-------|-----|----------|--------|
| NORMAL | 0.9383 | 0.9165 | 0.9611 |
| EARLY_WARNING | 0.7719 | 0.7765 | 0.7674 |
| ELEVATED | 0.6667 | 0.6914 | 0.6437 |
| CRITICAL | 0.6903 | 0.8125 | 0.6000 |

**Konfigurasi:**
- Ensemble type: Soft voting (probability averaging)
- Base models: Baseline BERT (001) + BERT+LSTM (002)
- Weights: Equal (1:1) - weight tuning skipped due to time constraints
- Device: NVIDIA GeForce RTX 4080 (CUDA)

**Comparison with Base Models:**
| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| Baseline BERT (001) | 64.82% | 0.4734 |
| BERT+LSTM (002) | 79.17% | 0.6589 |
| **Ensemble (003)** | **86.04%** | **0.7668** |

**Gains:**
- vs Baseline: +21.22% accuracy, +29.34% F1
- vs BERT+LSTM: +6.87% accuracy, +10.79% F1

**Key Findings:**
1. **Ensemble significantly outperforms individual models** - both models contribute complementary predictions
2. **CRITICAL class precision improved to 81.25%** - ensemble is more confident when predicting critical anomalies
3. **EARLY_WARNING F1 reached 0.77** - +11% over BERT+LSTM alone
4. All targets exceeded with simple 1:1 weighted average - no complex weight tuning needed
5. Ensemble reduces variance and improves confidence on borderline cases

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
- [x] Ensemble (001+002): 86.0% acc, 0.77 F1
- [ ] Hierarchical transformer

### Hyperparameter Learnings
- [x] Memory: RTX 4080 16GB - can handle larger batches, but used batch=8 for consistency
- [x] Window overlap: stride=5 (50%) better than non-overlapping
- [x] Early stopping: patience=4 appropriate (best at epoch 14)
- [x] Ensemble: Equal weights (1:1) work well, weight tuning not necessary for initial results

---

## Progress Checklist

### Phase 2: Core Development
- [x] Experiment 001: Baseline BERT
- [x] Experiment 002: BERT+LSTM
- [x] Experiment 003: Ensemble
- [x] Experiment 004: Hierarchical Transformer - Model implemented, ready to run on RTX 4080
- [ ] Experiment 005: Change Point Detection
- [ ] Ablation studies

---

## Today's Progress (2026-01-08)

### Experiment 004 - Hierarchical Transformer
- **Status:** Model implemented, ready to run
- **Model:** Hierarchical Transformer (BERT + Utterance-level Transformer)
- **Parameters:** 135M
- **Files created:**
  - `src/models/hierarchical_transformer.py` - Model implementation
  - `experiments/004_hierarchical/config.yaml` - Configuration
  - `experiments/004_hierarchical/run.py` - Training script with checkpoint support
  - `experiments/004_hierarchical/README.md` - Documentation
  - `experiments/004_hierarchical/QUICKSTART_OFFICE.md` - Quick start guide

### New Feature: Checkpoint & Resume
- **Auto-save:** Best model saved to `models/004/best_model.pt`
- **Resume capability:** Training can be resumed from checkpoint if interrupted
- **Early stopping:** Prevents overfitting, saves training time

### Next Steps
1. Run Exp 004 on RTX 4080 at office (~45-60 min)
2. Analyze results and compare with Ensemble (003)
3. If needed: Run Exp 005 (Change Point Detection) or ablation studies
