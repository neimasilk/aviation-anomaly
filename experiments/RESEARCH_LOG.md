# Research Log

**Catatan utama untuk research ini.** Baca ini dulu sebelum liat kode.

---

## Quick Reference: Apa yang Berhasil?

| # | Eksperimen | Status | Hasil Utama | Catatan |
|---|------------|--------|-------------|---------|
| 001 | Baseline BERT | Completed | Acc: 64.8%, F1: 0.47 | NORMAL class works well (79%), anomaly classes poor (36-38%) |
| 002 | BERT+LSTM | Completed | Acc: 79.2%, F1: 0.66 | **+14% accuracy, +19% F1 vs baseline** - Sequential context helps significantly |

---

## Eksperimen Terakhir

**Tanggal:** 2026-01-07
**Eksperimen:** 002 - BERT+LSTM Sequential Modeling
**Status:** Completed
**Tujuan:** Improve upon baseline by incorporating sequential dependencies with Bi-LSTM

**Hasil:**
- Test Accuracy: 79.17% (+14.35% vs baseline)
- Macro F1: 0.6589 (+18.55% vs baseline)
- Per-Class F1: NORMAL (0.90), EARLY_WARNING (0.66), ELEVATED (0.50), CRITICAL (0.57)

**Konfigurasi:**
- Window size: 10 utterances
- Stride: 5 (50% overlap)
- LSTM: 256 hidden, 2 layers, bidirectional
- Attention mechanism over utterances
- Training: 15 epochs, batch size 8, LR 2e-5

**Per-Class Comparison vs Baseline:**
| Class | Exp 002 F1 | Baseline F1 | Improvement |
|-------|-----------|-------------|-------------|
| NORMAL | 0.9049 | 0.7855 | +0.1194 |
| EARLY_WARNING | 0.6590 | 0.3585 | +0.3005 |
| ELEVATED | 0.5039 | 0.3720 | +0.1319 |
| CRITICAL | 0.5679 | 0.3777 | +0.1902 |

**Key Findings:**
1. Sequential context is crucial - 19% F1 improvement validates hypothesis
2. EARLY_WARNING detection improved most (+30%) - temporal patterns help distinguish early anomalies
3. All anomaly classes improved - Bi-LSTM captures transitions between states
4. NORMAL class also improved - better at distinguishing truly normal from pre-anomaly

**Previous Experiments:**

*001 - Baseline BERT (2026-01-07)*
- Test Accuracy: 64.82%
- Macro F1: 0.4734
- Static per-utterance classification
- Bias toward majority class

---

## Learning Log

### Dataset
- [x] Noort dataset: 172 cases, 21,626 utterances
- [x] Position-based temporal labeling implemented
- [x] Sequential windows: 4,190 sequences (window=10, stride=5)

### Model
- [x] Baseline BERT: 64.8% acc, 0.47 F1
- [x] BERT+LSTM: 79.2% acc, 0.66 F1
- [ ] Hierarchical transformer

### Hyperparameter Learnings
- [x] Memory: RTX 3060 Ti 8GB needs batch=8, max_utt=10, max_len=64
- [x] Window overlap: stride=5 (50%) better than non-overlapping
- [x] Early stopping: patience=4 appropriate (best at epoch 14)

---

## Progress Checklist

### Phase 2: Core Development
- [x] Experiment 002: BERT+LSTM
- [ ] Experiment 003: Hierarchical Transformer
- [ ] Experiment 004: Change Point Detection
- [ ] Ablation studies
