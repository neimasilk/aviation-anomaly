# Q1 Readiness Assessment: Aviation Anomaly Detection

**Date:** 2026-02-06  
**Assessor:** Critical Review  
**Target:** Scopus Q1 Journal (Safety Science / EAAI / Expert Systems with Applications)  
**Timeline:** 6-12 months  

---

## Executive Summary

| Dimension | Score | Status |
|-----------|-------|--------|
| **Novelty** | 7/10 | ✅ Sufficient for Q1 |
| **Technical Quality** | 6/10 | ⚠️ Needs improvement |
| **Data Quality** | 4/10 | ❌ Major concern |
| **Methodology** | 6/10 | ⚠️ Partial |
| **Writing/Presentation** | 5/10 | ⚠️ Needs rewrite |
| **Overall Q1 Readiness** | 5.5/10 | ⚠️ **CONDITIONAL GO** |

**Verdict:** Layak dilanjutkan **DENGAN SYARAT** perbaikan signifikan pada data dan metodologi.

---

## 1. NOVELTY & KONTRIBUSI ILMIAH (7/10)

### ✅ Strengths

1. **Gap yang Jelas dan Valid**
   - Sequential CVR analysis memang belum ada yang serius
   - Paper existing (BERT/RoBERTa untuk CVR) hanya static classification
   - Research gap jelas: "when does anomaly start?" vs "is this utterance anomalous?"

2. **Kontribusi Praktis**
   - Safety-critical domain (aviation)
   - Hasil bisa langsung dipakai untuk CRM training
   - Reduksi 91% missed detection adalah angka yang kuat

3. **Benchmark Komprehensif**
   - Multiple architectures dibandingkan (BERT, BERT-LSTM, Hierarchical, Ensemble)
   - Traditional baselines included (SVM 76%)
   - Statistical testing (McNemar's test)

### ⚠️ Weaknesses

1. **Novelty Arsitektural Terbatas**
   - BERT+LSTM bukan arsitektur baru (sudah mainstream 2019-2020)
   - Hierarchical Transformer gagal (overfitting)
   - Tidak ada arsitektur truly novel

2. **Kontribusi Teoritis Lemah**
   - Tidak ada linguistic theory yang di-validate
   - "Temporal markers" di-klaim tapi belum di-identify secara eksplisit
   - Tidak ada causal analysis

### 🎯 Q1 Requirements

Untuk Q1, Anda butuh **salah satu** dari:
- [ ] Arsitektur yang truly novel (contoh: custom attention mechanism untuk CVR)
- [x] Kontribusi teoritis kuat (linguistic patterns validated)
- [ ] Application yang completely new (CVR analysis memang baru)
- [ ] Performance yang state-of-the-art (86% acc bukan SOTA untuk 4-class)

**Status:** Cukup untuk Q1 Safety Science / EAAI, tapi bukan ACL/EMNLP tier-1.

---

## 2. KUALITAS DATA (4/10) ⚠️ KRISIS

### ❌ Critical Issues

1. **Position-Based Labeling = Artifisial**
   ```
   Masalah: Label didasarkan pada posisi dalam transcript, bukan konten.
   
   Contoh absurd:
   - "Mayday mayday mayday" di menit ke-11 sebelum crash = NORMAL
   - "Cleared for takeoff" di menit ke-0.5 sebelum crash = CRITICAL
   
   Impact: Model belajar "waktu", bukan "anomali komunikasi"
   ```

2. **Class Imbalance Ekstrem**
   ```
   NORMAL:        65.4% (14,136)
   EARLY_WARNING: 20.0% (4,326)
   ELEVATED:      10.0% (2,164)
   CRITICAL:       4.6% (1,000)  ← SEVERELY UNDERREPRESENTED
   
   Imbalance Ratio: 14.1:1
   
   Evidence: CRITICAL recall hanya 47-60% (model avoid prediksi class ini)
   ```

3. **Dataset Kecil untuk Deep Learning**
   ```
   Total utterances: 21,626
   Total sequences:  ~4,190 (setelah windowing)
   Train set:        ~2,933 sequences
   
   Untuk 135M parameters (Hierarchical Transformer): GROSSLY INSUFFICIENT
   ```

4. **No "Normal" Data**
   - Semua data dari CVR = kecelakaan
   - Tidak ada baseline "normal flight communication"
   - Model belajar "tingkat kepanikan", bukan "anomali vs normal"

### ⚠️ Moderate Issues

5. **Incomplete Cases**
   - 8 cases (4.7%) tidak ada CRITICAL labels
   - 2 cases (1.2%) tidak ada ELEVATED labels
   - Apakah ini incomplete transcripts atau accident type berbeda?

6. **Sequence Length Variation**
   ```
   Min: 1 utterance
   Max: 669 utterances
   Mean: 125.7 ± 133.6
   CV: 1.06 (extremely high)
   
   Problem: Statistical properties berbeda antara short dan long sequences
   ```

### 🎯 Rekomendasi Perbaikan Data

**Prioritas 1 (WAJIB untuk Q1):**
- [ ] **Content-Based Labeling:** Gunakan LLM untuk label berdasarkan konten, bukan waktu
- [ ] **LLM Annotation Pipeline:** Sudah dibuat (`scripts/annotation/llm_annotate.py`), jalankan!
- [ ] **Hybrid Labels:** Kombinasi position + content dengan confidence threshold

**Prioritas 2 (STRONGLY RECOMMENDED):**
- [ ] **Add Normal Data:** Scrape LiveATC.net atau gunakan simulated normal flight
- [ ] **SMOTE Augmentation:** Sudah di-run (Exp 006), tunggu hasil

**Prioritas 3 (NICE TO HAVE):**
- [ ] **Cross-validation by accident type:** Stratified K-fold
- [ ] **Temporal split:** Train on older cases, test on newer

---

## 3. METODOLOGI (6/10)

### ✅ Strengths

1. **Comprehensive Experimental Design**
   - 14 eksperimen direncanakan (001-014)
   - Multiple architectures
   - Ablation studies
   - Statistical testing

2. **Safety-Aware Evaluation**
   - Custom metrics: `safety_weighted_f1()`, `early_detection_score()`
   - Cost-sensitive matrix (miss CRITICAL = cost 20x)
   - Detection latency metrics

3. **Reproducibility**
   - Semua experiment pakai config YAML
   - Predictions disimpan (.npy)
   - Code well-structured

### ⚠️ Weaknesses

1. **Window Size Arbitrary**
   - Window=10, Stride=5 (kenapa?)
   - Belum di-validate optimal window size
   - Ablation study (Exp 008) masih pending

2. **Label dari Last Utterance**
   - Mengabaikan 9 utterances sebelumnya
   - Mungkin ada informasi penting di utterances awal window

3. **Evaluation Metrics Gap**
   - Belum ada time-to-detection analysis
   - Belum ada ROC-AUC per class
   - Belum ada calibration analysis

4. **No External Validation**
   - Semua data dari satu dataset (Noort et al.)
   - Tidak ada cross-dataset validation

### 🎯 Rekomendasi Metodologi

**WAJIB:**
- [ ] **Exp 009: Labeling Comparison** - Jalankan setelah LLM annotation selesai
- [ ] **Exp 014: K-Fold CV** - Non-negotiable untuk journal
- [ ] **Calibration Analysis** - Plot reliability diagram

**STRONGLY RECOMMENDED:**
- [ ] **Exp 008: Window Ablation** - Test window sizes [5, 10, 15, 20]
- [ ] **Feature Importance** - SHAP atau attention weights analysis

---

## 4. TECHNICAL QUALITY (6/10)

### ✅ Strengths

1. **Code Quality Bagus**
   - Well-structured (src/, experiments/, scripts/)
   - Modular design
   - Type hints, docstrings

2. **Infrastructure Lengkap**
   - Logging system
   - Checkpoint management
   - Google Drive sync
   - Git version control

### ⚠️ Weaknesses

1. **Hierarchical Transformer Gagal**
   - Underperform vs BERT-LSTM
   - Overfitting (val F1 0.70, test F1 0.61)
   - 135M parameters terlalu banyak untuk dataset kecil

2. **Exp 011 (DeBERTa) Error**
   - Float16 vs Float32 mismatch
   - Perlu fix sebelum bisa jalan

3. **Some Experiments Pending**
   - Exp 009, 012 butuh API key
   - Exp 013 butuh GPU 16GB (masih feasible)

---

## 5. WRITING & PRESENTATION (5/10)

### ⚠️ Issues

1. **Paper Draft Masih Template-like**
   - Abstract bagus, tapi sections lainnya generic
   - Discussion section lemah
   - Limitations belum di-address

2. **Referensi Kurang**
   - Hanya 4 referensi di .bib
   - Perlu 40+ referensi untuk Q1
   - Missing recent works (2023-2025)

3. **Figures Belum Final**
   - Placeholder images
   - Perlu professional visualization

### 🎯 Rekomendasi Writing

**Target Venue Analysis:**

| Venue | IF | Requirements | Fit |
|-------|-----|--------------|-----|
| **Safety Science** | ~6.0 | Safety-focused, rigorous methods | 8/10 ✅ BEST FIT |
| **EAAI** | ~8.0 | Engineering applications | 7/10 ✅ |
| **Expert Systems** | ~7.5 | Practical AI systems | 7/10 ✅ |
| **Applied Soft Computing** | ~8.7 | Hybrid systems | 6/10 |
| **IEEE Access** | ~3.4 | Fast review, lower bar | 9/10 (backup) |
| **ACL/EMNLP** | - | Novelty arsitektur tinggi | 4/10 ❌ |

**Rekomendasi Primary Target:**
1. **Safety Science** (Elsevier) - Best fit untuk domain
2. **EAAI** (Elsevier) - Good fit untuk methodology

---

## 6. RISK ANALYSIS

### High Risk (Bisa Gagal Total)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Content-based labeling tidak better | Medium | HIGH | Pivot ke "labeling comparison" sebagai kontribusi |
| SMOTE tidak improve CRITICAL recall | Medium | HIGH | Fokus pada cost-sensitive learning |
| Reviewer tolak karena dataset kecil | Medium | HIGH | Emphasize "pilot study" dan "foundation for future work" |

### Medium Risk (Manageable)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training takes too long | Low | Low | GPU rental atau Colab Pro |
| Exp 011-014 gagal | Medium | Low | Fokus pada 001-010 yang sudah solid |

---

## 7. GO / NO-GO DECISION

### ✅ GO - Jika Anda Bisa Commit pada:

1. **WAJIB (Non-negotiable):**
   - [ ] Run LLM annotation (Exp 009 prerequisite) - ~2-3 hari
   - [ ] Run Exp 009: Labeling Comparison - ~1-2 hari
   - [ ] Run Exp 014: K-Fold CV - ~1-2 hari GPU
   - [ ] Rewrite paper dengan focus pada "labeling matters" - ~1 minggu
   - [ ] Expand referensi ke 40+ - ~2-3 hari

2. **STRONGLY RECOMMENDED:**
   - [ ] Tunggu Exp 006 (SMOTE) selesai - ~2-4 jam
   - [ ] Run Exp 008: Window Ablation - ~1 hari GPU
   - [ ] Generate professional figures - ~2-3 hari

**Total Estimasi Waktu:** 3-4 minggu part-time (jika LLM annotation smooth)

### ❌ NO-GO - Jika:

- Anda tidak punya API key untuk LLM annotation
- Anda tidak bisa rewrite paper total
- Anda menargetkan ACL/EMNLP tier-1 (novelty tidak cukup)

---

## 8. RECOMMENDED ROADMAP TO Q1

### Phase 1: Data Fix (Week 1-2) 🚨 PRIORITAS TINGGI
```
1. Setup API key (DeepSeek/OpenAI)
2. Run LLM annotation (2-3 hari)
3. Validasi manual 200 sample (4-6 jam)
4. Run Exp 009: Labeling Comparison
5. Run Exp 014: K-Fold CV
```

### Phase 2: Experiments Completion (Week 3)
```
1. Tunggu Exp 006 selesai
2. Fix & run Exp 011 (DeBERTa)
3. Run Exp 008 (Window Ablation)
4. Statistical testing dengan semua results
```

### Phase 3: Paper Writing (Week 4-6)
```
1. Rewrite Introduction dengan "labeling as contribution"
2. Expand Related Work (40+ references)
3. Rewrite Methodology section
4. Add comprehensive Discussion
5. Generate professional figures
6. Internal review & iterate
```

### Phase 4: Submission (Week 7)
```
1. Target: Safety Science atau EAAI
2. Format according to journal guidelines
3. Submit and pray 🙏
```

---

## 9. FINAL VERDICT

### 🟡 CONDITIONAL GO

**Proyek ini LAYAK dilanjutkan ke Q1 dengan syarat:**

1. **Anda commit untuk LLM annotation** (kontribusi utama)
2. **Rewrite paper total** dengan focus pada "content-based labeling"
3. **Target realistic:** Safety Science / EAAI (bukan ACL/EMNLP)

**Success Probability:**
- Dengan perbaikan data: **70%** diterima di Safety Science
- Tanpa perbaikan data: **20%** diterima (kemungkinan major revision/reject)

**Kunci Sukses:**
> "The labeling strategy is the hero, not the model architecture."

Pivot cerita dari "sequential model bagus" ke "labeling by position adalah problem yang serius dan content-based labeling significantly better".

---

## 10. PERTANYAAN KRITIS UNTUK ANDA

Sebelum melanjutkan, jawab ini dengan jujur:

1. **Apakah Anda punya API key untuk LLM annotation?**
   - Jika TIDAK: Proyek akan stuck di Exp 009

2. **Apakah Anda bersedia rewrite paper total?**
   - Jika TIDAK: Paper akan ditolak karena lemah

3. **Apakah Anda punya waktu 3-4 minggu untuk push ini?**
   - Jika TIDAK: Consider lower-tier venue (Q2/Q3)

4. **Apakah Anda open untuk pivot cerita ke "labeling matters"?**
   - Jika TIDAK: Novelty arsitektur tidak cukup untuk Q1

Jika jawaban semua **YA**, maka GO.  
Jika ada **TIDAK**, maka perlu reconsider strategy.

---

**Assessment completed.**  
**Decision:** 🟡 **CONDITIONAL GO** - Continue with major data improvements.
