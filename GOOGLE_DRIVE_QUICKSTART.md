# Google Drive Setup - Quick Start

> Langkah-langkah setup Google Drive untuk storage data & model.

---

## Yang Perlu Dilakukan (3 Langkah)

### Langkah 1: Install Rclone

```bash
# Windows - pakai Chocolatey (recommended)
choco install rclone

# Atau download manual dari:
# https://rclone.org/downloads/

# Cek install
rclone version
```

### Langkah 2: Hubungkan ke Google Drive

```bash
# Jalankan konfigurasi
rclone config

# Follow prompts:
# - Type "n" (new remote)
# - Name: "gdrive"
# - Storage: "drive" (pilih nomor, biasanya 16)
# - scope: "1" (Full access)
# - root_folder_id: enter (kosongkan)
# - service_account_count: enter (kosongkan)
# - Edit advanced config: "n"
# - Use auto config: "Y" (penting!)
#   → Browser akan terbuka, login akun kampus
# - Configure shared drive: enter (kosongkan)
# - Keep token: "Y"
# - config ok: "y"
# - Quit config: "q"
```

### Langkah 3: Upload .env ke Drive (Satu Kali)

```bash
# Windows
.\scripts\sync_drive.bat upload-secrets

# Linux/Mac
chmod +x scripts/sync_drive.sh
./scripts/sync_drive.sh upload-secrets
```

---

## Selesai! Ini Cara Pakainya

### Di Komputer Baru (Pertama Kali)

```bash
# 1. Install rclone
# 2. rclone config (login)
# 3. Download secrets dari Drive
.\scripts\sync_drive.bat secrets

# 4. Download data
.\scripts\sync_drive.bat download
```

### Setiap Selesai Training/Preprocessing

```bash
# Upload hasil ke Drive
.\scripts\sync_drive.bat upload
```

### Sebelum Mulai Training di Komputer Lain

```bash
# Download data & model terbaru
.\scripts\sync_drive.bat download
```

---

## Struktur di Google Drive

Setelah sync, folder Drive akan jadi:

```
My Drive/
└── aviation-research/
    ├── datasets/          → Data files
    ├── models/            → Model checkpoints
    ├── outputs/           → Results, plots
    └── secrets/           → .env, API keys
```

---

## Troubleshooting

### Error: "Access token expired"

```bash
rclone config reconnect gdrive:
```

### Error: "rclone not found"

```bash
# Tambah rclone ke PATH, atau pakai full path
# Windows: C:\Program Files\Rclone\rclone.exe
```

### Browser tidak terbuka saat config

```bash
# Pakai manual config
rclone config
# Pilih: "n" → "gdrive" → "drive" → "1"
# Saat "Use auto config", pilih "n"
# Salin URL ke browser manually
```

---

## Summary

| Command | Kapan Pakai |
|---------|-------------|
| `rclone config` | Pertama setup |
| `sync_drive.bat secrets` | Komputer baru |
| `sync_drive.bat upload-secrets` | Setelah setup .env |
| `sync_drive.bat download` | Sebelum kerja |
| `sync_drive.bat upload` | Selesai kerja |
