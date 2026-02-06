# EXECUTION GUIDE: Baby-Step Panduan Eksekusi Riset

> Panduan ini ditulis agar junior developer atau model AI yang kurang capable
> bisa menjalankan setiap langkah tanpa perlu memahami keseluruhan sistem.
> Setiap step berdiri sendiri, punya input/output jelas, dan bisa diverifikasi.

---

## OVERVIEW: Apa yang Sudah Ada vs Belum

### FILE YANG SUDAH DIBUAT (kode siap jalan):

| File | Fungsi | Status |
|------|--------|--------|
| `src/evaluate/safety_metrics.py` | Metrik evaluasi safety-aware (EDS, Safety-F1, dll) | BARU |
| `scripts/annotation/llm_annotate.py` | Pipeline annotasi CVR pakai LLM | BARU |
| `src/data/preprocessing.py` | Ditambah `assign_content_based_labels()` & `assign_hybrid_labels()` | DIMODIFIKASI |
| `experiments/009_labeling_comparison/run.py` | Exp: bandingkan 3 strategi labeling | BARU |
| `experiments/010_traditional_baselines/run.py` | Exp: 9 baseline tradisional ML | ✅ **DONE** (Acc: 76.0%) |
| `experiments/011_deberta_lstm/run.py` | Exp: DeBERTa-v3 gantikan BERT | BARU |
| `experiments/012_llm_fewshot/run.py` | Exp: few-shot LLM classification | BARU |
| `experiments/013_finetuned_llm/run.py` | Exp: QLoRA fine-tune Phi-3-mini | BARU |
| `experiments/014_kfold_evaluation/run.py` | Exp: 5-fold cross-validation | BARU |
| `scripts/analysis/statistical_testing.py` | Ditambah `load_predictions()`, `bootstrap_confidence_interval()` | DIMODIFIKASI |
| `scripts/analysis/attention_visualization.py` | Visualisasi attention heatmap | BARU |
| `scripts/analysis/error_analysis.py` | Analisis error per subgrup | BARU |

### STATUS EKSEKUSI:

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 (Setup) | ✅ Done | Environment ready, dependencies installed |
| Phase 1 (Exp 006, 008) | ⏳ Ready | Kode verified, butuh waktu GPU |
| Phase 2 (Exp 010) | ✅ **DONE** | 9 baselines jalan, hasil tersimpan |
| Phase 3 (Exp 011-013) | ⏳ Ready | 011 & 013 butuh GPU, 012 butuh API key |
| Phase 4 (LLM Annotation) | ⏸️ Blocked | Butuh DeepSeek/OpenAI API key |
| Phase 5 (Exp 009) | ⏸️ Blocked | Butuh Phase 4 selesai |
| Phase 6 (Exp 014) | ⏳ Ready | K-Fold CV ready, butuh ~12 jam GPU |
| Phase 7 (Analysis) | ✅ Partial | Statistical testing ✅, viz pending |

**Catatan:** Exp 010 sudah berhasil dijalankan 2026-02-06. Hasil: TF-IDF + SVM best traditional baseline (Acc: 76.0%, F1: 0.635). Eksperimen lain siap jalan.

---

## PHASE 0: SETUP & PREREQUISITES

### Step 0.1: Cek Environment
```bash
# Pastikan di folder project
cd D:\documents\aviation-anomaly

# Cek Python & pip
python --version        # Harus 3.8+
pip list | grep torch   # Harus ada pytorch

# Cek data ada
ls data/cvr_labeled.csv
# ATAU
ls data/processed/cvr_transcripts.csv

# Cek GPU (opsional tapi sangat disarankan)
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 0.2: Install Dependencies Baru
```bash
# Wajib untuk Exp 010 (traditional baselines)
pip install xgboost sentence-transformers

# Wajib untuk Exp 013 (QLoRA fine-tuning)
pip install peft bitsandbytes trl accelerate

# Sudah harus ada dari sebelumnya:
# torch, transformers, scikit-learn, pandas, numpy, rich, tqdm, seaborn, matplotlib
```

### Step 0.3: Cek API Keys (untuk Exp 012 dan annotation)
```bash
# Buka file .env, pastikan ada:
# DEEPSEEK_API_KEY=sk-xxxxx
# OPENAI_API_KEY=sk-xxxxx  (opsional, untuk annotator kedua)

# Cek:
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('DeepSeek:', 'OK' if os.getenv('DEEPSEEK_API_KEY') else 'MISSING')"
```

---

## PHASE 1: JALANKAN EKSPERIMEN TERTUNDA [~1 hari GPU]

### Step 1.1: Run Exp 006 (SMOTE-Augmented) — SUDAH DI-FIX
```bash
cd experiments/006_smote_augmented
python run.py
```
- **Input**: `data/cvr_labeled.csv`
- **Output**: `outputs/experiments/006/results.json`
- **Verifikasi**: Buka results.json, cek `critical_recall` > 0.50
- **Estimasi waktu**: 2-4 jam GPU, 8-12 jam CPU
- **Jika error**: Coba `python run_simple.py` sebagai alternatif

### Step 1.2: Run Exp 008 (Window Size Ablation)
```bash
cd experiments/008_ablation_window_size
python run.py
```
- **Input**: `data/cvr_labeled.csv`
- **Output**: `outputs/experiments/008/results.json`
- **Verifikasi**: Buka results.json, harus ada 4 hasil (window 5, 10, 15, 20)
- **Estimasi**: 8-16 jam GPU (4 model dilatih)
- **Tips**: Bisa dijalankan paralel per window size di GPU berbeda

---

## PHASE 2: TRADITIONAL ML BASELINES [~30 menit CPU]

### Step 2.1: Run Exp 010
```bash
cd experiments/010_traditional_baselines
python run.py
```
- **Input**: `data/cvr_labeled.csv` (atau `data/processed/cvr_transcripts.csv`)
- **Output**:
  - `outputs/experiments/010/results.json` (semua metrik)
  - `outputs/experiments/010/y_true.npy` (ground truth)
  - `outputs/experiments/010/predictions_*.npy` (prediksi per model)
- **Verifikasi**: results.json harus punya 9 model results
- **Estimasi**: 10-30 menit (CPU only, kecuali Sentence-BERT ~10 menit GPU)

**Jika xgboost belum terinstall:**
```bash
pip install xgboost
# Lalu run ulang
```

**Jika sentence-transformers belum terinstall:**
- 8 model lainnya tetap jalan, Sentence-BERT di-skip otomatis
- Install nanti: `pip install sentence-transformers`

### Step 2.2: Verifikasi Output
```bash
python -c "
import json
with open('outputs/experiments/010/results.json') as f:
    r = json.load(f)
print(f'Models tested: {len(r[\"models\"])}')
for m in r['models']:
    print(f'  {m[\"name\"]}: Acc={m[\"metrics\"][\"accuracy\"]:.4f} F1={m[\"metrics\"][\"macro_f1\"]:.4f}')
"
```

---

## PHASE 3: FOUNDATION MODEL EXPERIMENTS [~1-3 hari]

### Step 3.1: Run Exp 011 (DeBERTa+LSTM) — BUTUH GPU
```bash
cd experiments/011_deberta_lstm
python run.py
```
- **Input**: `data/cvr_labeled.csv`
- **Output**: `outputs/experiments/011/results.json`, `models/011/best_model.pt`
- **Verifikasi**: Bandingkan F1 dengan Exp 002 (0.6589). Harusnya lebih tinggi.
- **Estimasi**: 2-4 jam di RTX 4080
- **Jika OOM**: Kurangi batch_size di config.yaml (4 → 2)

### Step 3.2: Run Exp 012 (Few-Shot LLM) — BUTUH API KEY
```bash
cd experiments/012_llm_fewshot

# Test dulu dengan sample kecil:
python run.py --max-samples 50

# Kalau OK, run full:
python run.py

# Opsional: test zero-shot juga:
python run.py --zero-shot --max-samples 100

# Opsional: test dengan OpenAI:
python run.py --provider openai --max-samples 100
```
- **Input**: API key di .env + data
- **Output**: `outputs/experiments/012/results_few_shot_deepseek.json`
- **Verifikasi**: Cek accuracy dan F1. LLM biasanya 40-65% accuracy pada task ini.
- **Estimasi**: ~$2-5 untuk full test set, ~10-30 menit
- **Jika rate limited**: Naikkan `--rate-limit 1.0` (1 detik delay)

### Step 3.3: Run Exp 013 (QLoRA Fine-Tune) — BUTUH GPU 16GB+
```bash
cd experiments/013_finetuned_llm
python run.py
```
- **Input**: `data/cvr_labeled.csv`
- **Output**: `outputs/experiments/013/results.json`, `models/013/best_adapter/`
- **Verifikasi**: F1 harus > 0.60. Bandingkan dengan Exp 002 & 011.
- **Estimasi**: 3-5 jam di RTX 4080 16GB
- **Jika OOM**: Edit config.yaml, kurangi `batch_size: 1`, naikkan `gradient_accumulation_steps: 16`

**PENTING: Jika tidak punya GPU 16GB, SKIP langkah ini.** Paper masih bisa submit tanpa Exp 013.

---

## PHASE 4: LLM ANNOTATION [~$5-15, ~2-4 jam]

### Step 4.1: Annotasi dengan DeepSeek
```bash
cd scripts/annotation

# Run annotation (akan memakan waktu ~2 jam untuk 21,626 utterances)
python llm_annotate.py --provider deepseek --rate-limit 0.3

# Progress disimpan otomatis setiap 500 utterances
# Jika terputus, resume:
python llm_annotate.py --provider deepseek --resume-from 5000  # ganti angka sesuai progress
```
- **Input**: `data/cvr_labeled.csv` + DeepSeek API key
- **Output**: `data/annotated/cvr_annotated_deepseek.csv`
- **Verifikasi**: File harus punya kolom `llm_score` (1-5) dan `llm_label`
- **Estimasi**: ~$2-5, ~1-2 jam

### Step 4.2: Annotasi dengan OpenAI (untuk inter-annotator agreement)
```bash
python llm_annotate.py --provider openai --model gpt-4o-mini --rate-limit 0.5
```
- **Output**: `data/annotated/cvr_annotated_openai.csv`
- **Estimasi**: ~$3-8

### Step 4.3: Hitung Inter-Annotator Agreement
```bash
python llm_annotate.py --compute-agreement
```
- **Output**: `data/annotated/inter_annotator_agreement.json`
- **Verifikasi**: `cohen_kappa_linear_weighted` harus > 0.60 (substantial agreement)

### Step 4.4: Buat Hybrid Labels
```bash
python llm_annotate.py --create-hybrid
```
- **Output**: `data/annotated/cvr_hybrid_labeled.csv`

### Step 4.5: Buat Sample untuk Validasi Manual
```bash
python llm_annotate.py --create-validation-sample
```
- **Output**: `data/annotated/manual_validation_sample.csv`
- **Action**: Buka CSV ini, isi kolom `manual_score` dan `manual_label` untuk 200 sample

---

## PHASE 5: LABELING COMPARISON EXPERIMENT [~6-12 jam GPU]

### Step 5.1: Run Exp 009
**PREREQUISITE**: Phase 4 harus selesai (annotasi LLM ada)

```bash
cd experiments/009_labeling_comparison

# Run semua 3 strategi:
python run.py

# ATAU run satu-satu (untuk debug / GPU terbatas):
python run.py --strategy position_based
python run.py --strategy content_based
python run.py --strategy hybrid
```
- **Input**: Data + annotations dari Phase 4
- **Output**: `outputs/experiments/009/results.json`
- **Verifikasi**: Bandingkan 3 strategi. Ekspektasi: hybrid > content > position untuk CRITICAL recall.
- **Estimasi**: 6-12 jam (3 model dilatih)

---

## PHASE 6: K-FOLD CROSS-VALIDATION [~12-24 jam GPU]

### Step 6.1: Run Exp 014
```bash
cd experiments/014_kfold_evaluation

# Run semua 5 fold:
python run.py

# ATAU run per fold (untuk 4 GPU paralel):
python run.py --fold 0  # GPU 1
python run.py --fold 1  # GPU 2
python run.py --fold 2  # GPU 3
python run.py --fold 3  # GPU 4
python run.py --fold 4  # GPU 1 (setelah fold 0 selesai)
```
- **Input**: `data/cvr_labeled.csv`
- **Output**:
  - `outputs/experiments/014/results.json`
  - `outputs/experiments/014/y_true_fold0.npy` ... `y_true_fold4.npy`
- **Verifikasi**: Di results.json, cek `summary.macro_f1.std` < 0.05
- **Estimasi**: ~12 jam (5 fold x ~2.5 jam)

---

## PHASE 7: STATISTICAL TESTING & ANALYSIS [~15 menit CPU]

### Step 7.1: Run Statistical Testing
**PREREQUISITE**: Minimal Exp 010 + salah satu DL experiment harus selesai

```bash
python scripts/analysis/statistical_testing.py
```
- **Output**: `outputs/analysis/statistical_results.json` + `model_comparison_table.csv`
- **Verifikasi**: p-values < 0.05 untuk klaim signifikansi

### Step 7.2: Attention Visualization
**PREREQUISITE**: Exp 002 atau 011 harus punya checkpoint

```bash
# Pakai model Exp 002:
python scripts/analysis/attention_visualization.py --model-path models/002/best_model.pt

# Atau pakai DeBERTa (Exp 011):
python scripts/analysis/attention_visualization.py --model-path models/011/best_model.pt --encoder microsoft/deberta-v3-base
```
- **Output**: `outputs/analysis/attention_maps/` (PNG files + JSON metadata)
- **Verifikasi**: Buka `attention_grid.png`, pastikan heatmap terlihat masuk akal

### Step 7.3: Error Analysis
```bash
# Analisis Exp 002:
python scripts/analysis/error_analysis.py --exp-id 002

# Analisis Exp 011:
python scripts/analysis/error_analysis.py --exp-id 011
```
- **Output**: `outputs/analysis/error_analysis/`
- **Verifikasi**: Buka confusion matrix PNG, cek JSON untuk insight

---

## PRIORITAS JIKA WAKTU TERBATAS

Jika hanya punya waktu terbatas, jalankan dalam urutan ini:

| Prioritas | Step | Waktu | Butuh GPU? | Butuh API? |
|-----------|------|-------|------------|------------|
| **1** | Step 2.1 (Traditional Baselines) | 30 menit | Tidak | Tidak |
| **2** | Step 1.1 (Exp 006 SMOTE) | 2-4 jam | Ya | Tidak |
| **3** | Step 3.1 (DeBERTa+LSTM) | 2-4 jam | Ya | Tidak |
| **4** | Step 4.1 (LLM Annotation DeepSeek) | 2 jam | Tidak | Ya ($2-5) |
| **5** | Step 6.1 (K-Fold CV) | 12 jam | Ya | Tidak |
| **6** | Step 3.2 (Few-Shot LLM) | 30 menit | Tidak | Ya ($2-5) |
| **7** | Step 7.1-7.3 (Analysis) | 15 menit | Tidak | Tidak |
| **8** | Step 5.1 (Labeling Comparison) | 6-12 jam | Ya | Tidak |

**MINIMUM VIABLE PAPER** = Prioritas 1-5 + 7 selesai.

---

## TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'src'"
```bash
# Pastikan di root project directory
cd D:\documents\aviation-anomaly
pip install -e .
```

### "CUDA out of memory"
- Edit `config.yaml` di folder experiment:
  - Kurangi `batch_size` (8 → 4 → 2)
  - Kurangi `max_utterance_length` (128 → 64)
  - Tambah `gradient_accumulation_steps` (biar effective batch tetap sama)

### "FileNotFoundError: data/cvr_labeled.csv"
- Data mungkin di `data/processed/cvr_transcripts.csv`
- Script otomatis coba kedua path, tapi jika tidak:
```bash
# Cek apa yang ada:
ls data/
ls data/processed/
ls data/raw/
```

### "API rate limit exceeded"
- Naikkan `--rate-limit` ke 1.0 atau 2.0 (detik)
- Atau jalankan di jam non-peak

### Experiment gagal di tengah jalan
- Kebanyakan experiment save checkpoint otomatis
- Cek folder `models/{exp_id}/` untuk checkpoint
- Exp 006: Bisa resume dari checkpoint yang tersimpan

---

## CHECKLIST FINAL SEBELUM SUBMIT PAPER

- [ ] Exp 010 jalan (Traditional Baselines) — WAJIB
- [ ] Minimal 1 DL experiment baru jalan (011 DeBERTa atau 006 SMOTE)
- [ ] K-Fold CV (Exp 014) selesai 5 fold — WAJIB untuk journal
- [ ] Statistical testing (Step 7.1) jalan dengan data real
- [ ] LLM Annotation (Phase 4) selesai — MAIN CONTRIBUTION
- [ ] Labeling comparison (Exp 009) selesai — MAIN CONTRIBUTION
- [ ] Attention visualization ada — NICE TO HAVE
- [ ] Error analysis ada — NICE TO HAVE
- [ ] 40+ referensi di paper — WAJIB
- [ ] paper.tex updated dengan semua hasil baru

---

## ARSITEKTUR FILE (setelah semua selesai)

```
aviation-anomaly/
├── data/
│   ├── cvr_labeled.csv                    # Data utama (sudah ada)
│   └── annotated/                         # BARU (Phase 4)
│       ├── cvr_annotated_deepseek.csv
│       ├── cvr_annotated_openai.csv
│       ├── cvr_hybrid_labeled.csv
│       ├── inter_annotator_agreement.json
│       └── manual_validation_sample.csv
│
├── experiments/
│   ├── 006_smote_augmented/               # Phase 1
│   ├── 008_ablation_window_size/          # Phase 1
│   ├── 009_labeling_comparison/           # Phase 5 (BARU)
│   ├── 010_traditional_baselines/         # Phase 2 (BARU)
│   ├── 011_deberta_lstm/                  # Phase 3 (BARU)
│   ├── 012_llm_fewshot/                   # Phase 3 (BARU)
│   ├── 013_finetuned_llm/                 # Phase 3 (BARU)
│   └── 014_kfold_evaluation/              # Phase 6 (BARU)
│
├── outputs/experiments/
│   ├── 010/results.json                   # Hasil baseline
│   ├── 011/results.json                   # Hasil DeBERTa
│   ├── 012/results_few_shot_deepseek.json # Hasil LLM
│   ├── 014/results.json                   # Hasil K-Fold
│   └── ...
│
├── outputs/analysis/
│   ├── statistical_results.json           # Statistical tests
│   ├── attention_maps/                    # Heatmaps (BARU)
│   └── error_analysis/                    # Error breakdown (BARU)
│
├── src/evaluate/
│   └── safety_metrics.py                  # Novel metrics (BARU)
│
├── scripts/annotation/
│   └── llm_annotate.py                    # Annotation pipeline (BARU)
│
└── EXECUTION_GUIDE.md                     # File ini
```
