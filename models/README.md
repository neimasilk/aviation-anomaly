# Models Directory

## Strategy: Local = Active Experiment Only, Google Drive = Full Backup

### Local Storage (SSD-friendly)
Only keep models for the **current/active experiment**:
- `models/004/` - Hierarchical Transformer (ACTIVE - training in progress)

### Google Drive (Full Backup)
All models are safely backed up to `aviation-research/models/`:
```
gdrive:aviation-research/models/
├── 001/best_model.pt     - Baseline BERT (418MB)
├── 002/best_model.pt     - BERT+LSTM (433MB)
└── 004/                  - Hierarchical Transformer (when ready)
```

### Downloading Specific Models from Drive

If you need a specific model locally (e.g., to run ensemble inference):

```bash
# Download only model 001
rclone copy gdrive:aviation-research/models/001 models/001

# Download only model 002
rclone copy gdrive:aviation-research/models/002 models/002

# Download all (NOT recommended - uses lots of SSD)
rclone copy gdrive:aviation-research/models/ models/
```

### After Experiment Completes

When an experiment is done and you move to the next one:

```bash
# 1. Upload to Drive (if not already)
./scripts/sync_drive.bat upload

# 2. Delete local model to save SSD
rm -rf models/00X

# 3. Start next experiment
# models/00Y will be created during training
```

### Model Sizes

| Experiment | Model | Size | Status |
|------------|-------|------|--------|
| 001 | Baseline BERT | 418MB | On Drive only |
| 002 | BERT+LSTM | 433MB | On Drive only |
| 003 | Ensemble | N/A | Uses 001+002, not saved separately |
| 004 | Hierarchical | TBD | Training (local only for now) |

---

**Rule:** Keep only what you're actively working on locally. Everything else on Drive.
