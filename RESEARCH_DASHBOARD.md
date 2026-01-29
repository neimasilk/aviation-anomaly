# Research Dashboard

> **Dashboard utama untuk tracking progress penelitian.**
> File ini adalah "single source of truth" - baca ini dulu sebelum apapun.

---

## Quick Status

| Aspect | Status | Notes |
|--------|--------|-------|
| **Google Drive** | ✅ Configured | Unlimited storage kampus |
| **Dataset** | ✅ Processed & Labeled | 21,626 utterances, 172 cases |
| **Baseline (001)** | ✅ Completed | Acc: 64.8%, F1: 0.47 |
| **Model A (BERT+LSTM)** | ✅ Completed | Acc: 79.2%, F1: 0.66 |
| **Model B (Hierarchical)** | ✅ Completed | Acc: 76.1%, F1: 0.61 - Overfitted |
| **Ensemble (003)** | ✅ Completed | **Acc: 86.0%, F1: 0.77** (Target exceeded!) |
| **Model C (Change Point)** | ✅ Completed | MAE: 49.1 utt, Early: 65.7% - Novel approach |
| **Paper** | ✅ Phase 3 | Statistical testing & visualization complete |

**Current Phase:** Paper Writing - All experiments completed
**Deadline:** On track for journal submission

---

## Dataset Summary

| File | Location | Size |
|------|----------|------|
| Raw (SPSS) | `raw/mmc4.sav` | 118 MB |
| Processed CSV | `cvr_transcripts.csv` | 3.0 MB |
| Labeled CSV | `cvr_labeled.csv` | 3.2 MB |

**Label Distribution (position-based):**
- NORMAL: 65.4% (early conversation, >10 min before)
- EARLY_WARNING: 20.0% (5-10 min before)
- ELEVATED: 10.0% (1-5 min before)
- CRITICAL: 4.6% (final minute)

---

## Google Drive Workflow

```
Drive Storage (aviation-research/)
├── datasets/    → Raw & processed data
├── models/      → Checkpoints
├── outputs/     → Results, plots
└── secrets/     → .env, API keys
```

### Commands

```bash
# Setup (pertama kali)
rclone config                    # Login akun kampus
.\scripts\sync_drive.bat secrets  # Download .env

# Daily workflow
.\scripts\sync_drive.bat download  # Sebelum mulai
# ... kerja ...
.\scripts\sync_drive.bat upload    # Selesai kerja
```

---

## Research Questions

1. **Kapan** anomali mulai terdeteksi sebelum kecelakaan?
2. Apakah sequential model > static classification?
3. Feature linguistik apa yang paling prediktif?
4. Bagaimana performa di berbagai time windows?

---

## Cara Pakai Repo Ini

### Setup Baru (Komputer Baru)

```bash
# 1. Clone & install
git clone <repo>
cd aviation-anomaly
pip install -r requirements.txt

# 2. Setup Google Drive
rclone config              # Login
.\scripts\sync_drive.bat secrets   # Download .env
.\scripts\sync_drive.bat download   # Download data
```

### Mulai Eksperimen Baru

```bash
# 1. Copy template
cp -r experiments/templates experiments/002_my_exp

# 2. Edit & run
cd experiments/002_my_exp
vim config.yaml
python run.py

# 3. Upload ke Drive
cd ../..
.\scripts\sync_drive.bat upload

# 4. Update log
vim experiments/RESEARCH_LOG.md
git add experiments/
git commit -m "exp: 002 progress"
git push
```

---

## Eksperimen Progress

### Completed

| # | Nama | Hasil | Conclusion |
|---|------|-------|------------|
| 001 | Baseline BERT | Acc: 64.8%, F1: 0.47 | Baseline established |
| 002 | BERT + LSTM | Acc: 79.2%, F1: 0.66 | Sequential modeling helps significantly |
| 003 | Ensemble | Acc: 86.0%, F1: 0.77 | **All targets exceeded!** |
| 004 | Hierarchical Transformer | Acc: 76.1%, F1: 0.61 | Underperformed - overfitting with 135M params |
| 005 | Change Point Detection | MAE: 49.1 utt, Early: 65.7% | **Novel approach** - detects anomaly onset time |

### Statistical Analysis

| Comparison | Test | Result |
|------------|------|--------|
| 001 vs 002 | McNemar's | **p < 0.001** - Sequential modeling significantly better |
| 001 vs 004 | McNemar's | **p < 0.001** - Hierarchical significantly better than baseline |

### In Progress

| # | Nama | Status |
|---|------|--------|
| Paper Writing | Journal Preparation | Statistical testing & figures complete |

### Deferred

| # | Nama | Reason |
|---|------|--------|
| 006 | Focal Loss | Target already achieved with Ensemble |
| 007 | Data Augmentation | Not needed - dataset sufficient |

---

## Quick Commands

```bash
# Sync dengan Google Drive
.\scripts\sync_drive.bat download    # Download data/model
.\scripts\sync_drive.bat upload      # Upload hasil
.\scripts\sync_drive.bat secrets     # Download .env

# Create new experiment
cp -r experiments/templates experiments/00X_name

# Run experiment 001
cd experiments/001_baseline_bert
python run.py

# Preprocess data
python -m src.data.preprocessing

# Git sync
git pull
git add experiments/ src/
git commit -m "update"
git push
```

---

## Quick Links

| File | Purpose |
|------|---------|
| **RESEARCH_DASHBOARD.md** | **This file** |
| **GOOGLE_DRIVE_QUICKSTART.md** | Drive setup guide |
| **experiments/RESEARCH_LOG.md** | Experiment log |
| **research_proposal.md** | Full proposal |
| **experiments/001_baseline_bert/** | First experiment |

---

## Next Steps

1. **Run 004** - `cd experiments/004_hierarchical && python run.py` (Hierarchical Transformer)
2. **Evaluate** - Compare with ensemble (003) results
3. **Upload to Drive** - `.\scripts\sync_drive.bat upload` (after each experiment)

---

## Recent Progress (Jan 2026)

- ✅ Google Drive setup with rclone
- ✅ Noort dataset acquired (mmc4.sav from Mendeley)
- ✅ SPSS → CSV conversion
- ✅ Position-based temporal labeling implemented
- ✅ Labeled dataset uploaded to Drive
- ✅ **Exp 001**: Baseline BERT - 64.8% acc, 0.47 F1
- ✅ **Exp 002**: BERT+LSTM - 79.2% acc, 0.66 F1
- ✅ **Exp 003**: Ensemble - **86.0% acc, 0.77 F1** (Target exceeded!)
