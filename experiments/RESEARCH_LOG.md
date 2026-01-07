# Research Log

**Catatan utama untuk research ini.** Baca ini dulu sebelum liat kode.

---

## Quick Reference: Apa yang Berhasil?

| # | Eksperimen | Status | Hasil Utama | Catatan |
|---|------------|--------|-------------|---------|
| 001 | Baseline BERT | ⏳ Pending | - | Static per-utterance classification |

---

## Eksperimen Terakhir

**Tanggal:** 2026-01-07
**Eksperimen:** 001 - Baseline BERT
**Status:** ⏳ Pending (waiting for dataset)
**Tujuan:** Establish baseline with static BERT (~80% accuracy expected)
**Hasil:** -
**Next Step:** Download Noort et al. (2021) dataset

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
- [x] Noort dataset structure identified
- [ ] Temporal labeling effectiveness
- [ ] Class distribution

### Model
- [x] BERT baseline architecture defined
- [ ] LSTM sequential modeling
- [ ] Hierarchical transformer

### Features
- [ ] Linguistic features importance
- [ ] Temporal patterns
- [ ] Aviation keywords

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
- [ ] Download Noort et al. dataset
- [ ] Exploratory data analysis
- [ ] Understand data structure
- [ ] Build preprocessing pipeline
- [ ] Run Experiment 001 (Baseline)
- [ ] Document baseline results

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
