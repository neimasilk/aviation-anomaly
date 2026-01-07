# Research Dashboard

> **Dashboard utama untuk tracking progress penelitian.**
> File ini adalah "single source of truth" - baca ini dulu sebelum apapun.

---

## 📍 Quick Status

| Aspect | Status | Notes |
|--------|--------|-------|
| **Dataset** | ⏳ Not acquired | Noort et al. (2021) - belum didownload |
| **Baseline** | ❌ Not started | Belum ada eksperimen |
| **Model A (BERT+LSTM)** | ❌ Not started | |
| **Model B (Hierarchical)** | ❌ Not started | |
| **Model C (Change Point)** | ❌ Not started | |
| **Paper** | ⏳ Phase 1 | Research proposal done |

**Current Phase:** Foundation (Dataset Acquisition)
**Deadline:** 6 months from Jan 2026

---

## 🎯 Research Questions Reminder

1. **Kapan** anomali mulai terdeteksi sebelum kecelakaan?
2. Apakah sequential model > static classification?
3. Feature linguistik apa yang paling prediktif?
4. Bagaimana performa di berbagai time windows?

---

## 🗂️ Cara Pakai Repo Ini

### Untuk Mulai Eksperimen Baru

```bash
# 1. Copy template
cp -r experiments/templates experiments/001_my_experiment

# 2. Rename & edit
cd experiments/001_my_experiment
# Edit config.yaml, run.py, README.md

# 3. Run
python run.py

# 4. Update RESEARCH_LOG.md
```

### Struktur Folder

```
experiments/
├── RESEARCH_LOG.md      # UPDATE INI SETELAPAH SETIAP EKSPERIMEN
├── templates/           # Template untuk eksperimen baru
├── 001_baseline_bert/   # Eksperimen yang selesai
├── 002_bert_lstm/       # Eksperimen yang selesai
├── 003_failed_xxx/      # Yang gagal - archive atau delete
└── archive/             # Eksperimen gagal tersimpan di sini

src/
├── core/                # Kode yang SUDAH TERBUKTI works
│   ├── data/           # Preprocessing yang verified
│   ├── models/         # Model architectures
│   └── utils/          # Utilities yang stable
└── experimental/       # Kode uji coba (bisa dihapus kalau gagal)
```

---

## 📊 Eksperimen Progress

### Completed

| # | Nama | Hasil | Conclusion |
|---|------|-------|------------|
| - | - | - | - |

### In Progress

| # | Nama | Status |
|---|------|--------|
| - | - | - |

### Queued

| # | Nama | Priority |
|---|------|----------|
| 001 | Baseline BERT | High |
| 002 | BERT + LSTM | High |
| 003 | Hierarchical Transformer | Medium |

---

## 💡 Key Insights So Far

*(Update section ini setelah learn sesuatu)*

### Dataset
- *No insights yet*

### Modeling
- *No insights yet*

### What Works
- *No insights yet*

### What Doesn't Work
- *No insights yet*

---

## 🔄 Pivot History

| Date | Decision | Reason |
|------|----------|--------|
| - | - | - |

---

## 🚨 Quick Commands

```bash
# Create new experiment
cp -r experiments/templates experiments/00X_name

# Run experiment
python experiments/00X_name/run.py

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Preprocess data
python -m src.core.data.preprocessing

# Train on training machine
git pull
python experiments/00X_name/run.py
git add experiments/00X_name/
git commit -m "exp: 00X results"
git push
```

---

## 📝 Documentation Checklist

Setiap eksperimen **WAJIB** punya:
- [ ] README.md dengan hasil & conclusion
- [ ] config.yaml dengan hyperparameters
- [ ] Metrics yang jelas
- [ ] Verdict (keep/discard/iterate)

---

## 🔗 Quick Links

- [Research Proposal](research_proposal.md) - Full proposal
- [RESEARCH_LOG](experiments/RESEARCH_LOG.md) - Detailed log
- [Claude Guide](CLAUDE.md) - Untuk AI assistant
- [README](README.md) - Project overview
