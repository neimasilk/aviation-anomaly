# Research Dashboard

> **Dashboard utama untuk tracking progress penelitian.**
> File ini adalah "single source of truth" - baca ini dulu sebelum apapun.

---

## Quick Status

| Aspect | Status | Notes |
|--------|--------|-------|
| **Dataset** | ⏳ Not acquired | Noort et al. (2021) - belum didownload |
| **Baseline (001)** | ⏳ Ready to run | Config, code, docs complete |
| **Model A (BERT+LSTM)** | ⏳ Queued | Waiting for baseline |
| **Model B (Hierarchical)** | ⏳ Queued | Waiting for baseline |
| **Model C (Change Point)** | ⏳ Queued | Waiting for baseline |
| **Paper** | ⏳ Phase 1 | Research proposal done |

**Current Phase:** Foundation (Dataset Acquisition)
**Deadline:** 6 months from Jan 2026

---

## Research Questions Reminder

1. **Kapan** anomali mulai terdeteksi sebelum kecelakaan?
2. Apakah sequential model > static classification?
3. Feature linguistik apa yang paling prediktif?
4. Bagaimana performa di berbagai time windows?

---

## Cara Pakai Repo Ini

### Mulai Eksperimen Baru

```bash
# 1. Copy template
cp -r experiments/templates experiments/002_my_exp

# 2. Rename & edit
cd experiments/002_my_exp
vim config.yaml  # Edit title, description, hyperparams
vim README.md     # Edit experiment overview

# 3. Run
python run.py

# 4. Update RESEARCH_LOG.md setelah selesai
```

### Struktur Folder

```
aviation-anomaly/
├── RESEARCH_DASHBOARD.md    # ⭐ BACA INI - Single source of truth
├── research_proposal.md     # Full proposal
│
├── experiments/             # 🧪 Semua eksperimen
│   ├── RESEARCH_LOG.md      # Update setelah selesai
│   ├── templates/           # Copy ini untuk baru
│   ├── 001_baseline_bert/   # ✅ Siap jalan (butuh data)
│   ├── 002_xxx/            # Next experiment
│   └── archive/             # Gagal → pindah sini
│
├── src/                     # ✅ Proven code only
│   ├── data/               # Preprocessing
│   ├── models/             # Model architectures
│   └── utils/              # Config, DeepSeek API
│
├── data/                   # Not in git
├── models/                 # Not in git
└── logs/                   # Not in git
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

## Key Insights So Far

### Dataset
- Noort et al. (2021) dataset identified
- 172 transcripts, 21,626 utterances
- Temporal labeling scheme defined

### Modeling
- Baseline architecture selected (BERT)
- Sequential models designed (BERT+LSTM, Hierarchical)

### What Works
- *No experiments run yet*

### What Doesn't Work
- *No experiments run yet*

---

## Pivot History

| Date | Decision | Reason |
|------|----------|--------|
| 2026-01-07 | Restructured for trial-and-error | Better handle failed experiments |

---

## Quick Commands

```bash
# Create new experiment
cp -r experiments/templates experiments/00X_name

# Run experiment 001
cd experiments/001_baseline_bert
python run.py

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Preprocess data (after download)
python -m src.data.preprocessing

# Multi-computer sync
git pull
# ... do work ...
git add experiments/ src/
git commit -m "update: progress"
git push
```

---

## Documentation Checklist

Setiap eksperimen **WAJIB** punya:
- [x] README.md dengan hasil & conclusion
- [x] config.yaml dengan hyperparameters
- [x] run.py yang executable
- [ ] Metrics yang jelas
- [ ] Verdict (keep/discard/iterate)

---

## Quick Links

| File | Purpose |
|------|---------|
| **RESEARCH_DASHBOARD.md** | **This file - status & progress** |
| **experiments/RESEARCH_LOG.md** | Detailed experiment log |
| **research_proposal.md** | Full research proposal |
| **experiments/001_baseline_bert/** | First experiment (ready) |
| **CLAUDE.md** | Guide for AI assistant |
| **README.md** | Project overview |

---

## Next Steps

1. **Download Dataset** - Noort et al. (2021) from Mendeley
2. **Preprocess** - Run `python -m src.data.preprocessing`
3. **Run 001** - `cd experiments/001_baseline_bert && python run.py`
4. **Evaluate** - Check if baseline matches ~80% accuracy
5. **Iterate** - If good, proceed to 002 (BERT+LSTM)
