# Quick Start - Office Computer

## Langkah Pertama (Setup)

```bash
# 1. Pull repo terbaru
cd D:/projects/aviation-anomaly
git pull origin main

# 2. Cek GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 3. Install dependencies (kalau belum)
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn pyyaml rich -q
```

## Jalankan Eksperimen

### Opsi 1: Ensemble (Paling Cepat - 1-2 jam)
```bash
python -m experiments.003_ensemble.run
```

### Opsi 2: Hierarchical Transformer (4-6 jam)
```bash
python -m experiments.004_hierarchical.run
```

### Opsi 3: Focal Loss (2-3 jam)
```bash
python -m experiments.005_focal_loss.run
```

## Cek Progress

```bash
# Lihat file yang berubah
git status

# Cek training log (kalau ada)
tail -f logs/*/training.log

# Cek GPU usage
nvidia-smi -l 1
```

## Setelah Selesai

```bash
# 1. Commit hasil
git add experiments/
git commit -m "exp: 003 ensemble completed"

# 2. Push ke GitHub
git push origin main

# 3. Upload model ke Drive
rclone copy models/003 gdrive:aviation-research/models/003 -P
rclone copy outputs/experiments/003 gdrive:aviation-research/outputs/003 -P
```

## Troubleshooting

### CUDA Out of Memory
- Edit `config.yaml`: kurangi `batch_size` (8 → 4)
- Kurangi `max_utterances` (10 → 8)

### Data Not Found
- Jalankan: `python scripts/add_temporal_labels.py`

### Import Error
- Cek working directory harus di root project
- Pastikan `src/` ada di PYTHONPATH

---

**File Referensi:**
- `ROADMAP.md` - Rencana lengkap eksperimen
- `experiments/RESEARCH_LOG.md` - Log eksperimen
