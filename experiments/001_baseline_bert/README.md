# Experiment 001: Baseline BERT

**Status:** ⏳ Pending
**Created:** 2026-01-07
**Completed:** -

---

## Overview

**Tujuan:** Mendirikan baseline performance dengan static BERT classification.

**Hipotesis:** BERT akan mencapai ~80% accuracy seperti yang dilaporkan di paper existing (RoBERTa untuk CVR sentiment analysis).

**Motivasi:** Perlu baseline untuk membandingkan dengan model sequential (BERT+LSTM) yang akan dikembangkan nanti.

---

## Setup

### Data
- Source: Noort et al. (2021) CVR Transcripts
- Preprocessing: Temporal labeling berdasarkan time-before-crash
- Split: 70/15/15 train/val/test by case_id

### Model
- Architecture: Static BERT (bert-base-uncased)
- Classification: Per-utterance (bukan sequential)
- Output: 4 classes (NORMAL, EARLY_WARNING, ELEVATED, CRITICAL)

### Hyperparameters
- Learning rate: 2e-5
- Batch size: 32
- Epochs: 10
- Dropout: 0.3

---

## Results

*(Update setelah run)*

### Metrics

| Metric | Expected | Actual | Notes |
|--------|----------|--------|-------|
| Accuracy | ~0.80 | - | |
| Macro F1 | ~0.75 | - | |
| CRITICAL Recall | - | - | Penting untuk safety |

---

## Analysis

### Apa yang Berhasil?

*(Setelah run)*

### Apa yang Tidak Berhasil?

*(Setelah run)*

### Insights

*(Setelah run)*

---

## Conclusion

**Verdict:** ⏳ Pending

**Action Items:**
- [ ] Download dataset
- [ ] Run preprocessing
- [ ] Train model
- [ ] Evaluate results

**Next Steps:**
- Setelah baseline tercapai → lanjut ke Experiment 002 (BERT+LSTM)

---

## Artifacts

- Config: `experiments/001_baseline_bert/config.yaml`
- Code: `experiments/001_baseline_bert/run.py`
- Checkpoint: `models/001/`
- Logs: `logs/001/`
- Results: `outputs/experiments/001/results.json`

---

## Notes

- Ini adalah reproduksi dari existing work
- Hasil akan menjadi baseline untuk eksperimen sequential
- Kalau hasil jauh dari 80%, investigasi preprocessing
