# Experiment {XXX}: {TITLE}

**Status:** 🔄 In Progress | ✅ Success | ❌ Failed | 📁 Archived
**Created:** {DATE}
**Completed:** {DATE}

---

## Overview

**Tujuan:** {Apa yang mau dicapai?}

**Hipotesis:** {Apa yang diuji?}

**Motivasi:** {Kenapa eksperimen ini penting?}

---

## Setup

### Data
- Source: {data yang dipakai}
- Preprocessing: {langkah preprocessing}
- Split: {train/val/test ratio}

### Model
- Architecture: {model type}
- Hyperparameters:
  - Learning rate: {value}
  - Batch size: {value}
  - Epochs: {value}
  - Other: {...}

### Training
- Device: {cpu/cuda}
- Time per epoch: {duration}
- Total training time: {duration}

---

## Results

### Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | - | - |
| Macro F1 | - | - |
| CRITICAL Recall | - | Penting untuk safety |
| EDS (Early Detection) | - | Custom metric |

### Confusion Matrix

*(Copy-paste atau screenshot)*

### Learning Curves

*(Catat apakah overfit/underfit)*

---

## Analysis

### Apa yang Berhasil?

*(Bagian yang works well)*

### Apa yang Tidak Berhasil?

*(Bagian yang bermasalah)*

### Insights

*(Apa yang dipelajari?)*

---

## Conclusion

**Verdict:** ✅ Keep | ❌ Discard | 🔄 Iterate

**Action Items:**
- [ ] {Next step 1}
- [ ] {Next step 2}

**Related Issues:** {Link ke issues atau eksperimen lain}

---

## Artifacts

- Config: `experiments/{XXX}/config.yaml`
- Code: `experiments/{XXX}/run.py`
- Checkpoint: `models/{XXX}/`
- Logs: `logs/{XXX}/`
- Plots: `outputs/{XXX}/`
