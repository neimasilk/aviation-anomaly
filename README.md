# Aviation Anomaly Detection

**Temporal Dynamics of Pilot Communication Before Aviation Accidents: A Sequence-Based Anomaly Detection Approach**

Research by Mukhlis Amien (STIKI Malang, 2026)

---

## 🎯 Quick Start

**Baca file ini dulu:** [RESEARCH_DASHBOARD.md](RESEARCH_DASHBOARD.md)

Dashboard berisi:
- Quick status
- Eksperimen progress
- Key insights
- Quick commands

---

## 📁 Project Structure

```
aviation-anomaly/
├── RESEARCH_DASHBOARD.md    # ⭐ BACA INI DULU - Single source of truth
├── research_proposal.md     # Full research proposal
│
├── experiments/             # 🧪 Semua eksperimen live here
│   ├── RESEARCH_LOG.md      # Detailed experiment log
│   ├── templates/           # Template for new experiments
│   ├── 001_baseline/        # Experiments (numbered)
│   ├── 002_xxx/
│   └── archive/             # Failed experiments
│
├── src/
│   ├── core/                # ✅ Kode yang SUDAH TERBUKTI works
│   │   ├── data/           # Preprocessing
│   │   ├── models/         # Model architectures
│   │   └── utils/          # Utilities
│   └── experimental/       # 🧪 Kode uji coba (bisahapus)
│
├── data/
│   ├── raw/                # Original dataset (not in git)
│   └── processed/          # Cleaned data (not in git)
│
├── models/                 # Trained models (not in git)
├── logs/                   # Training logs (not in git)
├── outputs/                # Plots, results (not in git)
│
├── .env                    # API keys (not in git)
├── .env.example            # Template untuk .env
├── config/default.yaml     # Default configuration
└── requirements.txt        # Dependencies
```

---

## 🚀 Quick Commands

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env dengan DeepSeek API key
```

### Create New Experiment

```bash
# 1. Copy template
cp -r experiments/templates experiments/001_my_exp

# 2. Edit files
cd experiments/001_my_exp
vim config.yaml
vim README.md

# 3. Run
python run.py

# 4. Update logs
vim ../RESEARCH_LOG.md
```

### Multi-Computer Workflow

```bash
# Regular computer - development
git pull
# ... make changes ...
git add experiments/ src/core/
git commit -m "update: experiment 001 results"
git push

# Training computer - GPU work
git pull
python experiments/001_my_exp/run.py
# ... git hanya track docs, bukan large files ...
```

---

## 📊 Research Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Foundation | 🔄 In Progress | Dataset acquisition, preprocessing |
| 2. Core Development | ⏳ Queued | Model implementation |
| 3. Analysis | ⏳ Queued | Results, paper writing |

See [RESEARCH_DASHBOARD.md](RESEARCH_DASHBOARD.md) for detailed status.

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| [RESEARCH_DASHBOARD.md](RESEARCH_DASHBOARD.md) | **Main dashboard** - status, progress, insights |
| [experiments/RESEARCH_LOG.md](experiments/RESEARCH_LOG.md) | Detailed experiment log |
| [research_proposal.md](research_proposal.md) | Full research proposal |
| [CLAUDE.md](CLAUDE.md) | Guide for AI assistant |

---

## 🧪 Experiment Template

Setiap eksperimen WAJIB punya:

```
experiments/00X_name/
├── README.md       # Hasil, conclusion, what worked/failed
├── config.yaml     # Hyperparameters
├── run.py          # Code to run
└── outputs/        # Plots, logs (not in git)
```

Use `experiments/templates/` as starting point.

---

## 📚 Dataset

**Primary:** Noort et al. (2021) CVR Transcript Dataset
- 172 unique transcripts (1962-2018)
- 21,626 lines of dialogue
- [DOI: 10.1016/j.dib.2021.107602](https://doi.org/10.1016/j.dib.2021.107602)

---

## 🔧 Tech Stack

- Python 3.8+
- PyTorch, Hugging Face Transformers
- Pandas, NumPy, Scikit-learn
- DeepSeek API (data augmentation)

---

## 📝 License

MIT License

---

## 🙏 Citation

```bibtex
@misc{amien2026aviation,
  title={Temporal Dynamics of Pilot Communication Before Aviation Accidents:
         A Sequence-Based Anomaly Detection Approach Using Transformer Models},
  author={Amien, Mukhlis},
  year={2026},
  institution={STIKI Malang}
}
```
