# Checklist Besok - Office PC (RTX 4080)

## 📋 Sebelum Berangkat

- [ ] Pastikan bawa laptop/power bank (kalau perlu tunggu)
- [ ] Cek list eksperimen yang akan dijalankan: **Exp 004 - Hierarchical Transformer**

---

## 🚀 Saat Sampai di Kantor

### 1. Sync & Setup (5 menit)

```bash
# Masuk ke project directory
cd D:\document\aviation-anomaly

# Sync dari Google Drive (opsional - pakai download-data untuk hemat SSD)
.\scripts\sync_drive.bat download-data
```

**Catatan:** Jangan pakai `download` biasa karena akan download semua model (851MB). Pakai `download-data` untuk skip model.

### 2. Cek Environment (2 menit)

```bash
# Cek GPU
nvidia-smi

# Pastikan CUDA available
python -c "import torch; print(torch.cuda.is_available())"
```

**Expected output:** `True` dan GPU RTX 4080 terdeteksi.

### 3. Run Experiment 004 (45-60 menit)

```bash
cd experiments/004_hierarchical
python run.py
```

**Bisa tinggal** - training akan berjalan sendiri.

### 4. Monitor Progress (Opsional)

Training akan menampilkan progress per epoch:
- Setiap epoch ~2-3 menit
- Early stopping setelah 5 epoch tanpa improvement
- Checkpoint auto-save setiap epoch

### 5. Setelah Selesai (5 menit)

```bash
# Upload hasil ke Google Drive
cd ../..
.\scripts\sync_drive.bat upload

# Push ke GitHub (opsional)
git add outputs/experiments/004/ models/004/
git commit -m "exp: 004 Hierarchical Transformer completed"
git push origin main
```

---

## 📊 Expected Results

| Metric | Target | Baseline (003) |
|--------|--------|----------------|
| Accuracy | >= 0.80 | 0.8604 |
| Macro F1 | >= 0.70 | 0.7668 |

**Ideal:** Hierarchical Transformer ~ensemble atau lebih baik.

---

## ⚠️ Jika Ter-interupsi

Training bisa di-resume kapan saja:

```bash
# Cukup jalankan lagi
cd experiments/004_hierarchical
python run.py

# Script otomatis detect checkpoint dan resume
```

**Checkpoint file:** `models/004/checkpoint.pt`

---

## 🧹 Cleanup Setelah Selesai

Setelah eksperimen selesai dan hasil di-upload:

```bash
# Hapus model lokal untuk hemat SSD (opsional - sudah aman di Drive)
rm -rf models/004
```

---

## 📝 Quick Reference Commands

```bash
# Sync (hemat SSD)
.\scripts\sync_drive.bat download-data
.\scripts\sync_drive.bat upload

# Run experiment
cd experiments/004_hierarchical
python run.py

# Cek GPU
nvidia-smi

# Download specific model dari Drive (kalau butuh)
rclone copy gdrive:aviation-research/models/001 models/001
rclone copy gdrive:aviation-research/models/002 models/002
```

---

## 🎯 Success Criteria

- [ ] Experiment 004 selesai tanpa error
- [ ] Results tersimpan di `outputs/experiments/004/results.json`
- [ ] Model tersimpan di `models/004/best_model.pt`
- [ ] Hasil di-upload ke Google Drive
- [ ] (Opsional) Commit dan push ke GitHub

---

## 📱 Screenshot untuk Dokumentasi (Opsional)

Ambil screenshot:
1. Training progress (epoch terbaik)
2. Final test results
3. Comparison table dengan previous experiments

---

**Created:** 2026-01-08
**Estimasi waktu total:** ~60-75 menit (termasuk setup & upload)
