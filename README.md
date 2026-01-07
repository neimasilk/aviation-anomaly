# Aviation Anomaly Detection

**Temporal Dynamics of Pilot Communication Before Aviation Accidents: A Sequence-Based Anomaly Detection Approach**

Research by Mukhlis Amien (STIKI Malang, 2026)

---

## Quick Start

**Baca file ini dulu:** [RESEARCH_DASHBOARD.md](RESEARCH_DASHBOARD.md)

---

## Project Structure

```
aviation-anomaly/
├── RESEARCH_DASHBOARD.md    # ⭐ Status & progress
├── GOOGLE_DRIVE_QUICKSTART.md # Google Drive setup
├── research_proposal.md     # Full proposal
│
├── experiments/             # All experiments
│   ├── RESEARCH_LOG.md
│   ├── templates/
│   ├── 001_baseline_bert/
│   └── archive/
│
├── src/                     # Core code
│   ├── data/
│   ├── models/
│   └── utils/
│
├── data/                   # Dari Drive (not in git)
├── models/                 # Dari Drive (not in git)
├── scripts/                # Utility scripts
│   └── sync_drive.bat      # Sync ke Google Drive
│
└── .env                    # Dari Drive (not in git)
```

---

## Setup (Pertama Kali)

### 1. Clone & Install

```bash
git clone <repo-url>
cd aviation-anomaly
pip install -r requirements.txt
```

### 2. Setup Google Drive (Storage Cloud)

```bash
# Install rclone
choco install rclone  # Windows

# Setup koneksi ke Google Drive
rclone config
# (follow prompts, login akun kampus)

# Download .env dari Drive
.\scripts\sync_drive.bat secrets

# Download data dari Drive
.\scripts\sync_drive.bat download
```

Lihat [GOOGLE_DRIVE_QUICKSTART.md](GOOGLE_DRIVE_QUICKSTART.md) untuk detail.

---

## Workflow dengan Google Drive

```
┌─────────────────┐     sync          ┌─────────────────┐
│  Komputer A     │ ────────────────> │  Google Drive   │
│  (Development)  │                   │  (Cloud Storage)│
└─────────────────┘                   └─────────────────┘
                                               │
                                               │ sync
                                               ↓
┌─────────────────┐                   ┌─────────────────┐
│  Komputer B     │ <──────────────── │   (Unlimited)   │
│  (Training GPU) │     download      │                 │
└─────────────────┘                   └─────────────────┘
```

### Commands

```bash
# Download data/model dari Drive (sebelum mulai)
.\scripts\sync_drive.bat download

# Upload hasil ke Drive (selesai kerja)
.\scripts\sync_drive.bat upload

# Download secrets (.env) di komputer baru
.\scripts\sync_drive.bat secrets
```

---

## Create New Experiment

```bash
# 1. Copy template
cp -r experiments/templates experiments/002_my_exp

# 2. Edit
cd experiments/002_my_exp
vim config.yaml
vim README.md

# 3. Run
python run.py

# 4. Upload hasil ke Drive
cd ../..
.\scripts\sync_drive.bat upload

# 5. Update log & commit
vim experiments/RESEARCH_LOG.md
git add experiments/
git commit -m "exp: 002 results"
git push
```

---

## Documentation

| File | Purpose |
|------|---------|
| [RESEARCH_DASHBOARD.md](RESEARCH_DASHBOARD.md) | Main dashboard |
| [experiments/RESEARCH_LOG.md](experiments/RESEARCH_LOG.md) | Experiment log |
| [GOOGLE_DRIVE_QUICKSTART.md](GOOGLE_DRIVE_QUICKSTART.md) | Drive setup |
| [research_proposal.md](research_proposal.md) | Full proposal |

---

## Dataset

**Primary:** Noort et al. (2021) CVR Transcript Dataset
- 172 unique transcripts (1962-2018)
- 21,626 lines of dialogue
- [DOI: 10.1016/j.dib.2021.107602](https://doi.org/10.1016/j.dib.2021.107602)

---

## Citation

```bibtex
@misc{amien2026aviation,
  title={Temporal Dynamics of Pilot Communication Before Aviation Accidents:
         A Sequence-Based Anomaly Detection Approach Using Transformer Models},
  author={Amien, Mukhlis},
  year={2026},
  institution={STIKI Malang}
}
```
