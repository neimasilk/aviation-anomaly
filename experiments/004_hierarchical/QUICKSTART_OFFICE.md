# Quick Start: Experiment 004 di Kantor (RTX 4080)

## Status
- **Experiment 004 files created** ✅
- **Hierarchical Transformer model implemented** ✅
- **Config ready** ✅
- **Data ready** ✅ (4,190 sequences)
- **Checkpoint/Resume feature** ✅ Added for safety

---

## Langkah untuk besok di kantor:

### 1. Sync dari Google Drive
```bash
cd D:\document\aviation-anomaly
.\scripts\sync_drive.bat download
```

### 2. Run Experiment 004
```bash
cd experiments/004_hierarchical
python run.py
```

**Resume jika ter-interupsi:**
```bash
# Jika training berhenti (misal: mati lampu, time limit), cukup jalankan lagi:
python run.py
# Script akan otomatis detect checkpoint dan resume dari epoch terakhir
```

### 3. Setelah selesai, upload hasil
```bash
cd ../..
.\scripts\sync_drive.bat upload
```

---

## Perkiraan waktu di RTX 4080:

| Activity | Estimasi |
|----------|----------|
| Setup & data loading | ~1 menit |
| Training per epoch | ~2-3 menit |
| Total training (dengan early stopping) | **~45-60 menit** |

**Early stopping:** Training berhenti otomatis jika tidak ada improvement setelah 5 epoch.

---

## Fitur Checkpoint & Resume:

### Auto-save (setiap epoch):
- **checkpoint.pt** - Full state (model + optimizer + scheduler + history)
- **best_model.pt** - Model dengan F1 terbaik

### Resume:
- Jika training ter-interupsi, jalankan `python run.py` lagi
- Script otomatis detect checkpoint dan resume
- Tidak perlu konfigurasi tambahan

### Manual restart dari awal:
```bash
# Hapus checkpoint jika ingin mulai dari awal
rm models/004/checkpoint.pt
```

---

## Konfigurasi:
- **Model:** Hierarchical Transformer
- **Parameters:** 135M
- **Batch size:** 8
- **Learning rate:** 1e-5
- **Max epochs:** 30 (patience=5)
- **Early stopping:** 5 epoch tanpa improvement

---

## Target hasil:
- Accuracy: >= 0.80 (baseline Ensemble: 0.8604)
- Macro F1: >= 0.70 (baseline Ensemble: 0.7668)

---

## Output files:
- `outputs/experiments/004/results.json` - Final metrics
- `models/004/best_model.pt` - Model terbaik
- `models/004/checkpoint.pt` - Checkpoint untuk resume
- `outputs/experiments/004/confusion_matrix.npy` - Confusion matrix

---

**Created:** 2026-01-08
**Next run:** Office PC with RTX 4080
