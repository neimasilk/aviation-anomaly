# Research Log

**Catatan utama untuk research ini.** Baca ini dulu sebelum liat kode.

---

## Quick Reference: Apa yang Berhasil?

| # | Eksperimen | Status | Hasil Utama | Catatan |
|---|------------|--------|-------------|---------|
| 001 | Baseline BERT | ✅ Completed | Acc: 64.8%, F1: 0.47 | NORMAL class works well (79%), anomaly classes poor (36-38%) |

---

## Eksperimen Terakhir

**Tanggal:** 2026-01-07
**Eksperimen:** 001 - Baseline BERT
**Status:** ✅ Completed
**Tujuan:** Establish baseline with static BERT (~80% accuracy expected)

**Hasil:**
- Test Accuracy: 64.82%
- Macro F1: 0.4734
- Per-Class F1: NORMAL (0.79), EARLY_WARNING (0.36), ELEVATED (0.37), CRITICAL (0.38)

**Confusion Matrix Analysis:**
- NORMAL: 81% recall (good) - model learns majority class well
- EARLY_WARNING: 36% recall - 54% misclassified as NORMAL
- ELEVATED: 34% recall - 43% misclassified as NORMAL
- CRITICAL: 29% recall - only 29% detected, but 53% precision when predicted

**Key Findings:**
1. Static BERT biased toward majority class (NORMAL = 65% of data)
2. Position-based temporal labeling creates detectable signal
3. Anomaly classes need sequential modeling (context matters)
4. Class weights alone insufficient for imbalanced temporal data

**Next Step:** Experiment 002 - BERT+LSTM for sequential patterns

---

## Pivot History

Catatan perubahan arah research yang signifikan:

| Tanggal | Dari | Ke | Alasan |
|---------|------|-----|--------|
| - | - | - | - |

---

## Learning Log

Hal-hal yang dipelajari (baik berhasil maupun gagal):

### Dataset
- [x] Noort dataset structure identified (172 cases, 21,626 utterances)
- [x] Position-based temporal labeling implemented
- [x] Class distribution: NORMAL (65%), EARLY_WARNING (20%), ELEVATED (10%), CRITICAL (5%)

### Model
- [x] BERT baseline architecture implemented
- [x] Baseline achieves 64.8% accuracy, 0.47 macro F1
- [ ] LSTM sequential modeling
- [ ] Hierarchical transformer

### Features
- [ ] Linguistic features importance
- [x] Position-based signal is detectable (not random)
- [ ] Temporal patterns (need sequential model)

---

## Dead Ends / Gagal

Eksperimen yang **TIDAK** perlu diulang:

| # | Nama | Kenapa Gagal |
|---|------|--------------|
| - | - | - |

---

## Rules untuk Eksperimen Baru

1. **Selalu pakai template** di `experiments/templates/`
2. **Numbering berurutan** (001, 002, 003, ...)
3. **Update log ini** setelah run selesai
4. **Jangan modify src/** kecuali sudah proven
5. **Archive kalau gagal** - pindah ke `experiments/archive/`

---

## Progress Checklist

### Phase 1: Foundation (Bulan 1-2)
- [x] Download Noort et al. dataset (mmc4.sav from Mendeley)
- [x] Exploratory data analysis
- [x] Understand data structure
- [x] Build preprocessing pipeline
- [x] Run Experiment 001 (Baseline)
- [x] Document baseline results
- [x] Setup Google Drive sync with rclone

### Phase 2: Core Development (Bulan 3-4)
- [ ] Experiment 002: BERT+LSTM
- [ ] Experiment 003: Hierarchical Transformer
- [ ] Experiment 004: Change Point Detection
- [ ] Ablation studies

### Phase 3: Analysis & Writing (Bulan 5-6)
- [ ] Error analysis
- [ ] Case studies
- [ ] Paper writing
- [ ] Submission
