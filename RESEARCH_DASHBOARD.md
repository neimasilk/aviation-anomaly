# Research Dashboard

> **Dashboard utama untuk tracking progress penelitian.**
> File ini adalah "single source of truth" - baca ini dulu sebelum apapun.

---

## Quick Status

| Aspect | Status | Notes |
|--------|--------|-------|
| **Google Drive** | ⏳ Ready to setup | Unlimited storage kampus |
| **Dataset** | ⏳ Not acquired | Noort et al. (2021) |
| **Baseline (001)** | ⏳ Ready to run | Config complete |
| **Model A (BERT+LSTM)** | ⏳ Queued | Waiting for baseline |
| **Model B (Hierarchical)** | ⏳ Queued | Waiting for baseline |
| **Model C (Change Point)** | ⏳ Queued | Waiting for baseline |
| **Paper** | ⏳ Phase 1 | Proposal done |

**Current Phase:** Foundation (Setup & Dataset)
**Deadline:** 6 months from Jan 2026

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
| - | - | - | - |

### In Progress

| # | Nama | Status |
|---|------|--------|
| 001 | Baseline BERT | ⏳ Ready (waiting for data) |

### Queued

| # | Nama | Priority |
|---|------|----------|
| 002 | BERT + LSTM | High |
| 003 | Hierarchical Transformer | Medium |
| 004 | Change Point Detection | Medium |

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

1. **Setup Google Drive** - Install rclone, run `rclone config`
2. **Download Dataset** - Noort et al. (2021)
3. **Preprocess** - `python -m src.data.preprocessing`
4. **Run 001** - `cd experiments/001_baseline_bert && python run.py`
5. **Upload to Drive** - `.\scripts\sync_drive.bat upload`
