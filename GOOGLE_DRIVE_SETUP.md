# Google Drive Setup Guide

> Menggunakan Google Drive unlimited kampus untuk central storage data & model.

---

## Konsep

```
┌─────────────────────────────────────────────────────────────┐
│                    GOOGLE DRIVE (Cloud)                     │
│  /aviation-research/                                        │
│    ├── datasets/          → Raw & processed data            │
│    ├── models/            → Trained model checkpoints       │
│    ├── outputs/           → Plots, results, logs            │
│    └── backups/           → Experiment archives             │
└─────────────────────────────────────────────────────────────┘
         ↑                    ↓                    ↑
    rclone upload       rclone download      Colab mount
  (local → Drive)     (Drive → local)     (GPU training)
         ↓                    ↓                    ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Komputer   │    │  Komputer   │    │   Google    │
│  Biasa      │    │  Training   │    │   Colab     │
│  (dev)      │    │  (GPU)      │    │   (free)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## Opsi 1: Rclone (Recommended)

**Rclone** = rsync untuk cloud. Sync file lokal ↔ Google Drive.

### Install Rclone

```bash
# Windows (chocolatey)
choco install rclone

# Atau download manual
# https://rclone.org/downloads/

# Cek install
rclone version
```

### Setup Google Drive

```bash
# 1. Konfigurasi Google Drive
rclone config

# Pilih:
# - New remote → nama: "gdrive"
# - Storage → "drive" (Google Drive)
# - scope → "1" (Full access all files)
# - root_folder_id → biarkan kosong (default)
# - service_account_file → biarkan kosong
# - advanced config → enter/enter
# - config_verification → enter/enter
# - auto config → "Y" (akan buka browser)
# - Login pakai akun kampus
# - Choose shared drive → biarkan kosong
# - Keep token → "Y"

# 2. Test koneksi
rclone ls gdrive:

# 3. Buat folder di Drive
rclone mkdir gdrive:aviation-research
rclone mkdir gdrive:aviation-research/datasets
rclone mkdir gdrive:aviation-research/models
rclone mkdir gdrive:aviation-research/outputs
```

### Sync Commands

```bash
# Upload data ke Drive
rclone copy data/ gdrive:aviation-research/datasets/ -P

# Download data dari Drive
rclone copy gdrive:aviation-research/datasets/ data/ -P

# Upload model checkpoints
rclone copy models/ gdrive:aviation-research/models/ -P

# Sync dua arah (hati-hati!)
rclone sync gdrive:aviation-research/datasets/ data/ -P
```

---

## Opsi 2: PyDrive (Python)

**PyDrive** = Python library untuk Google Drive API.

### Install

```bash
pip install PyDrive
```

### Setup (First Time Only)

```bash
# 1. Buat project di Google Cloud Console
#    https://console.cloud.google.com/
#    - New Project → "aviation-research"
#    - API & Services → Enable "Google Drive API"
#
# 2. Buat OAuth credentials
#    - API & Services → Credentials
#    - Create Credentials → OAuth client ID
#    - Application type → Desktop app
#    - Download JSON → simpan sebagai client_secrets.json
#
# 3. Copy client_secrets.json ke project
cp ~/Downloads/client_secrets.json aviation-anomaly/config/
```

### Install ke requirements.txt

```bash
pip install PyDrive
```

---

## Struktur Folder di Drive

```
aviation-research/ (root di Drive)
├── datasets/
│   ├── raw/                    → Noort dataset (original)
│   ├── processed/              → Preprocessed CSV
│   └── augmented/              → Synthetic data dari DeepSeek
│
├── models/
│   ├── 001_baseline_bert/      → Checkpoints exp 001
│   ├── 002_bert_lstm/          → Checkpoints exp 002
│   └── best/                   → Model terbaik tiap jenis
│
├── outputs/
│   ├── experiments/
│   │   ├── 001/               → Results, plots
│   │   └── 002/
│   └── figures/                → Figures untuk paper
│
└── backups/
    └── archive/                → Eksperimen gagal
```

---

## Workflow dengan Drive

### Komputer Biasa (Development)

```bash
# Download dataset dari Drive
rclone copy gdrive:aviation-research/datasets/ data/ -P

# ... coding, preprocessing ...

# Upload hasil preprocessing
rclone copy data/processed/ gdrive:aviation-research/datasets/processed/ -P
```

### Komputer Training (GPU)

```bash
# Download data yang sudah dipreprocess
rclone copy gdrive:aviation-research/datasets/processed/ data/processed/ -P

# ... training ...

# Upload model checkpoints
rclone copy models/ gdrive:aviation-research/models/ -P
```

### Google Colab (Alternatif GPU)

```python
from google.colab import drive
drive.mount('/content/drive')

# Link ke folder aviation-research
import os
os.chdir('/content/drive/MyDrive/aviation-research')

# Download dataset dari Drive (kalau perlu)
# !rclone copy gdrive:aviation-research/datasets/ /content/data -P

# ... training ...

# Auto save ke Drive (langsung ke Drive folder)
```

---

## Quick Script: Sync ke Drive

Buat file `scripts/sync_to_drive.sh`:

```bash
#!/bin/bash
# sync_to_drive.sh - Upload ke Google Drive

echo "🚀 Syncing to Google Drive..."

# Upload data
echo "📦 Uploading data..."
rclone copy data/ gdrive:aviation-research/datasets/ \
  --exclude="**/.gitkeep" \
  --exclude="**/.DS_Store" \
  --progress

# Upload models
echo "🤖 Uploading models..."
rclone copy models/ gdrive:aviation-research/models/ \
  --exclude="**/.gitkeep" \
  --exclude="**/.DS_Store" \
  --progress

# Upload outputs
echo "📊 Uploading outputs..."
rclone copy outputs/ gdrive:aviation-research/outputs/ \
  --exclude="**/.gitkeep" \
  --exclude="**/.DS_Store" \
  --progress

echo "✅ Sync complete!"
```

### Download dari Drive

Buat file `scripts/sync_from_drive.sh`:

```bash
#!/bin/bash
# sync_from_drive.sh - Download dari Google Drive

echo "🚥 Syncing from Google Drive..."

# Download data
echo "📦 Downloading data..."
rclone copy gdrive:aviation-research/datasets/ data/ \
  --exclude="**/.gitkeep" \
  --exclude="**/.DS_Store" \
  --progress

# Download models (opsional, kalau perlu resume)
echo "🤖 Downloading models..."
rclone copy gdrive:aviation-research/models/ models/ \
  --exclude="**/.gitkeep" \
  --exclude="**/.DS_Store" \
  --progress

echo "✅ Sync complete!"
```

---

## Tips

1. **Sync selective** - Jangan sync semua, pilih folder yang perlu saja
2. **Exclude patterns** - Gunakan --exclude untuk skip file besar yang tidak perlu
3. **Check space** - `rclone about gdrive:` untuk cek space
4. **Bandwidth** - Kampus mungkin limit bandwidth, jangan download/upload terlalu sering
5. **Versioning** - Drive punya version history, useful untuk rollback

---

## Troubleshooting

### Error: "Access token expired"

```bash
rclone config reconnect gdrive:
```

### Error: "Rate limit"

```bash
# Tambah delay antar file
rclone copy ... --bwlimit 10M  # limit 10 MB/s
```

### File besar gagal upload

```bash
# Increase chunk size
rclone copy ... --drive-chunk-size 64M
```
