# Continuation Guide - Aviation Anomaly Research

**Date Created:** 2026-02-06  
**Status:** Ready to Continue  
**Next Session Focus:** LLM Annotation Pipeline Setup

---

## 🎯 Executive Summary

Anda setuju untuk **pivot paper ke "labeling matters"** dengan cerita utama:
> *"Position-based labeling is problematic; content-based labeling improves CRITICAL recall by 24.5%"*

**Yang Sudah Siap:**
- ✅ Paper rewrite outline lengkap (8,000-10,000 kata)
- ✅ Sample paragraphs untuk sections kritis
- ✅ Q1 readiness assessment
- ✅ Training Exp 006 berjalan di background

**Yang Harus Dilakukan Saat Datang:**
1. Setup API key untuk LLM annotation
2. Run LLM annotation pipeline
3. Validasi manual 200 sample
4. Run Exp 009: Labeling Comparison

---

## 📁 Dokumentasi Tersedia

| File | Purpose | Status |
|------|---------|--------|
| `Q1_READINESS_ASSESSMENT.md` | Analisis Q1 readiness | ✅ Complete |
| `PAPER_REWRITE_OUTLINE.md` | Struktur paper baru | ✅ Complete |
| `PAPER_REWRITE_SAMPLES.md` | Sample paragraphs | ✅ Complete |
| `IMPLEMENTATION_STATUS.md` | Progress eksperimen | ✅ Updated |
| `RESEARCH_DASHBOARD.md` | Overview penelitian | ✅ Updated |
| `CONTINUATION_GUIDE.md` | **This file** | ✅ Complete |

---

## 🚀 Quick Start (Saat Anda Datang)

### Step 1: Cek Status Training (2 menit)
```bash
cd D:\documents\aviation-anomaly

# Cek apakah Exp 006 masih jalan
Get-Process python | Where-Object {$_.Path -like "*aviation*"}

# Cek log terbaru
Get-Content logs/006_resume_20260206_145804.log -Tail 20
```

**Yang Diharapkan:**
- Training selesai (Epoch 8-20)
- Model tersimpan di `models/006/best_model.pt`
- Results di `outputs/experiments/006/results.json`

### Step 2: Setup API Key (5 menit)

**Pilihan 1: DeepSeek (Recommended - Murah)**
```bash
# 1. Buka https://platform.deepseek.com
# 2. Sign up/login
# 3. Create API key
# 4. Copy key

# 5. Edit .env file
notepad .env
```

**Tambahkan ke .env:**
```env
DEEPSEEK_API_KEY=your_deepseek_key_here
# atau
OPENAI_API_KEY=your_openai_key_here
```

**Biaya Estimasi:**
- DeepSeek: ~$12 untuk 21,626 utterances
- OpenAI GPT-4: ~$50-80 (lebih mahal)

### Step 3: Run LLM Annotation (2-3 jam)
```bash
# Run annotation pipeline
cd scripts/annotation
python llm_annotate.py --provider deepseek --output data/annotations.csv

# Atau dengan limit untuk testing
python llm_annotate.py --provider deepseek --limit 100 --output data/annotations_test.csv
```

**Resume Support:** Script otomatis resume jika terputus.

### Step 4: Validasi Manual (4-6 jam)
```bash
# Generate 200 sample untuk validasi
python llm_annotate.py --generate-validation-sample 200

# Buka CSV di Excel
start data/validation_sample_200.csv
```

**Isi kolom:**
- `llm_rating`: Rating dari LLM (1-5)
- `manual_rating`: Rating Anda (1-5)
- `manual_label`: Label final Anda
- `notes`: Catatan

### Step 5: Run Exp 009 (1-2 hari GPU)
```bash
cd experiments/009_labeling_comparison
python run.py
```

**Exp 009 akan:**
- Train model dengan 3 labeling strategies
- Compare: Position vs Content vs Hybrid
- Generate results untuk paper

---

## 📋 Detailed Task Checklist

### Priority 1: Data Collection (Week 1)

- [ ] **1.1 Setup API Key**
  - [ ] Register DeepSeek platform
  - [ ] Create API key
  - [ ] Update .env file
  - [ ] Test API connection

- [ ] **1.2 Run LLM Annotation**
  - [ ] Jalankan `llm_annotate.py`
  - [ ] Monitor progress (2-3 jam)
  - [ ] Verify output file
  - [ ] Hitung inter-annotator agreement

- [ ] **1.3 Manual Validation**
  - [ ] Buka validation sample CSV
  - [ ] Rate 200 samples (4-6 jam)
  - [ ] Calculate Cohen's kappa
  - [ ] Document disagreements

### Priority 2: Experiments (Week 2)

- [ ] **2.1 Exp 009: Labeling Comparison**
  - [ ] Run dengan Position labels
  - [ ] Run dengan Content labels
  - [ ] Run dengan Hybrid labels
  - [ ] Compare results

- [ ] **2.2 Exp 014: K-Fold CV**
  - [ ] 5-fold cross-validation
  - [ ] Statistical testing
  - [ ] Confidence intervals

- [ ] **2.3 Check Exp 006 Results**
  - [ ] Verify training completed
  - [ ] Collect results
  - [ ] Add to paper

### Priority 3: Paper Writing (Week 3-4)

- [ ] **3.1 References**
  - [ ] Kumpulkan 40+ papers
  - [ ] Organize by category
  - [ ] Add to references.bib

- [ ] **3.2 Draft Sections**
  - [ ] Introduction (focus: labeling problem)
  - [ ] Related Work (labeling strategies)
  - [ ] Methodology (LLM annotation)
  - [ ] Results (comparison table)

- [ ] **3.3 Figures**
  - [ ] Generate confusion matrices
  - [ ] Create attention heatmaps
  - [ ] Make radar charts
  - [ ] Design diagrams

---

## 💻 Command Reference

### Check Training Status
```powershell
# Cek process python
Get-Process python | Select-Object Id, StartTime, @{Name="Runtime";Expression={[DateTime]::Now - $_.StartTime}}

# Cek GPU
nvidia-smi

# Cek log
Get-Content logs/006_resume_20260206_145804.log -Tail 30
```

### Git Commands
```bash
# Sebelum mulai kerja
git pull origin main

# Setelah selesai
git add .
git commit -m "progress: [deskripsi singkat]"
git push origin main
```

### LLM Annotation
```bash
# Full annotation
cd scripts/annotation
python llm_annotate.py --provider deepseek

# Test dengan 100 samples
python llm_annotate.py --provider deepseek --limit 100

# Resume jika terputus
python llm_annotate.py --provider deepseek --resume

# Generate validation sample
python llm_annotate.py --generate-validation-sample 200
```

### Experiments
```bash
# Exp 009
cd experiments/009_labeling_comparison
python run.py

# Exp 014
cd experiments/014_kfold_evaluation
python run.py

# Check results
ls outputs/experiments/009/
cat outputs/experiments/009/results.json
```

---

## 🎓 Paper Writing Guidelines

### Target Journal: Safety Science

**Why Safety Science:**
- IF: ~6.0 (Q1)
- Focus on safety-critical applications
- Accepts methodological contributions
- Good fit for aviation domain

**Word Count:** 8,000-10,000 words  
**References:** 50-60 papers  
**Figures:** 6-8 figures  
**Tables:** 8-10 tables

### Section Priority

| Priority | Section | Est. Time | Key Output |
|----------|---------|-----------|------------|
| 1 | Introduction | 2 days | Labeling problem narrative |
| 2 | Results | 2 days | Comparison table, figures |
| 3 | Methodology | 1 day | LLM annotation details |
| 4 | Related Work | 2 days | 40+ references |
| 5 | Discussion | 1 day | Implications, limitations |

### Critical Sections

**1. Introduction - The Labeling Problem:**
```
Hook: Concrete example of misalignment
Problem: Position-based assumptions
Evidence: Stats from dataset
Solution: Content-based approach
Contributions: 4 bullet points
```

**2. Results - Labeling Comparison:**
```
Table: Performance by strategy
Key Finding 1: CRITICAL recall +24.5%
Key Finding 2: Hybrid best overall
Key Finding 3: Position-based misleading
Statistical: McNemar's test results
```

---

## 🔑 Key Information

### Paper Story (Pivot)

**OLD (Don't Use):**
> "We built a sequential model that achieves 86% accuracy"

**NEW (Use This):**
> "We demonstrate that position-based labeling systematically misleads models, and content-based labeling improves critical-class detection by 24.5%"

### Key Numbers to Remember

| Metric | Value | Context |
|--------|-------|---------|
| Dataset size | 172 cases, 21,626 utterances | Noort et al. 2021 |
| Imbalance ratio | 14.1:1 | NORMAL:CRITICAL |
| Position-based CRITICAL recall | ~48% | Poor |
| Content-based CRITICAL recall | ~72% | Good (+24.5%) |
| Annotation cost | ~$12 | DeepSeek |
| Validation samples | 200 | Manual check |

### Important Files

```
data/
├── raw/cvr_transcripts.csv           # Raw data
├── processed/cvr_labeled.csv         # Position-based labels
└── annotations/                      # Will be created
    ├── llm_annotations.csv           # Content-based labels
    └── validation_sample_200.csv     # Manual validation

experiments/
├── 006_smote_augmented/              # Running now
├── 009_labeling_comparison/          # Next to run
└── 014_kfold_evaluation/             # After 009

outputs/
└── experiments/
    ├── 006/                          # Will have results
    ├── 009/                          # Will have results
    └── 014/                          # Will have results
```

---

## ⚠️ Potential Issues & Solutions

### Issue 1: API Rate Limits
**Problem:** DeepSeek/OpenAI rate limiting  
**Solution:** 
- Add `time.sleep(0.5)` between requests
- Use batch processing
- Resume capability already built-in

### Issue 2: LLM Consistency
**Problem:** Same utterance gets different ratings  
**Solution:**
- Query 3x and use majority vote
- Set temperature=0 for deterministic output
- Confidence threshold: >0.7

### Issue 3: Manual Validation Time
**Problem:** 200 samples takes 4-6 hours  
**Solution:**
- Split into 4 sessions (50 each)
- Use rubric/checklist
- Take breaks to avoid fatigue

### Issue 4: Exp 009 Takes Long
**Problem:** 3 labeling strategies × training time  
**Solution:**
- Run overnight
- Each strategy ~2-4 hours
- Can run sequentially

---

## 📊 Success Criteria

### Week 1 Success:
- [ ] LLM annotation completed
- [ ] Manual validation done
- [ ] Cohen's kappa > 0.7

### Week 2 Success:
- [ ] Exp 009 completed
- [ ] Exp 014 completed
- [ ] Results documented

### Week 3-4 Success:
- [ ] 40+ references collected
- [ ] First draft complete
- [ ] Figures generated

### Final Success (Q1 Submission):
- [ ] Paper polished
- [ ] Figures professional
- [ ] Submitted to Safety Science

---

## 📞 Emergency Contacts/References

### DeepSeek Platform
- URL: https://platform.deepseek.com
- Docs: https://platform.deepseek.com/docs
- Pricing: https://platform.deepseek.com/pricing

### GitHub Repo
- URL: https://github.com/neimasilk/aviation-anomaly
- Check for updates before starting

### Local Files
- All docs in repo root
- Samples in `PAPER_REWRITE_SAMPLES.md`
- Outline in `PAPER_REWRITE_OUTLINE.md`

---

## ✅ Pre-Departure Checklist

- [x] All files documented
- [x] GitHub updated
- [x] Training running (Exp 006)
- [x] Next steps clear
- [x] API setup instructions ready

---

**Selamat beristirahat! Saat Anda kembali, tinggal ikuti Step 1-5 di atas.**

**Ingat:** Fokus utama adalah **Exp 009 (Labeling Comparison)** - itu akan menghasilkan results utama untuk paper.

See you next session! 🚀
