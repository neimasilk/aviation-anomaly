# Research Roadmap

**Project:** Temporal Dynamics of Pilot Communication Before Aviation Accidents
**Researcher:** Mukhlis Amien (STIKI Malang, January 2026)
**Status:** Phase 2 - Core Development

---

## Progress Summary

### Completed (Phase 1)
- [x] Dataset acquisition (Noort et al., 2021)
- [x] Data preprocessing & temporal labeling
- [x] Experiment 001: Baseline BERT (Acc: 64.8%, F1: 0.47)
- [x] Experiment 002: BERT+LSTM (Acc: 79.2%, F1: 0.66)
- [x] Error analysis & documentation

### Key Findings So Far
1. Sequential modeling reduces missed detections by **91%** (534 → 46)
2. EARLY_WARNING recall improved **31%** with sequential context
3. CRITICAL precision high (71.9%) but recall low (46.9%)
4. ELEVATED remains hardest class (49.2% recall)

---

## Phase 2: Core Development (Priority Order)

### Experiment 003: Ensemble Baseline + Sequential
**Status:** Pending
**Expected:** +5-7% F1 improvement
**Effort:** LOW (1-2 hours)

**Rationale:**
- Exp 001 and Exp 002 have complementary error patterns
- Quick win using existing models

**Approach:**
1. Load both trained models
2. Soft voting with probability averaging
3. Test on validation set to find optimal weights
4. Evaluate on test set

**Commands:**
```bash
# From project root
python -m experiments.003_ensemble.run
```

**Files to create:**
- `experiments/003_ensemble/config.yaml`
- `experiments/003_ensemble/run.py`
- `experiments/003_ensemble/README.md`

---

### Experiment 004: Hierarchical Transformer
**Status:** Pending
**Expected:** +3-5% F1 improvement
**Effort:** MEDIUM (4-6 hours)

**Rationale:**
- LSTM may not capture all long-range dependencies
- Transformer better handles sequential attention
- Could improve ELEVATED/CRITICAL boundary detection

**Architecture:**
```
Input (sequence of utterances)
         ↓
BERT Encoder (token-level) - frozen or fine-tuned
         ↓
[CLS] tokens → Utterance embeddings
         ↓
Utterance-level Transformer (4-8 layers)
         ↓
Classification Head → 4-class prediction
```

**Hyperparameters to explore:**
- Transformer layers: 2, 4, 6, 8
- Hidden size: 128, 256, 512
- Attention heads: 4, 8
- Dropout: 0.1, 0.3, 0.5
- Learning rate: 1e-4, 5e-5, 1e-5

**Commands:**
```bash
python -m experiments.004_hierarchical.run
```

**GPU Requirements:**
- VRAM: 8-16 GB recommended
- Training time: ~2-4 hours

---

### Experiment 005: Focal Loss + Threshold Optimization
**Status:** Pending
**Expected:** +2-4% F1 improvement
**Effort:** LOW (2-3 hours)

**Rationale:**
- Current class weights may be suboptimal
- Focal loss focuses on hard examples
- Per-class threshold tuning can improve CRITICAL recall

**Focal Loss Formula:**
```
FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
γ = 2 (focusing parameter)
α = [0.25, 0.25, 0.25, 0.25] (class weights)
```

**Threshold Optimization:**
- Grid search on validation set
- Optimize for: F1, or custom safety metric
- Store optimal thresholds per class

**Commands:**
```bash
python -m experiments.005_focal_loss.run
```

---

### Experiment 006: Temporal Data Augmentation
**Status:** Pending
**Expected:** +2-3% F1 (especially CRITICAL)
**Effort:** MEDIUM (3-4 hours)

**Rationale:**
- CRITICAL class underrepresented (7.8% of data)
- Synthetic sequences can improve learning

**Augmentation Strategies:**

1. **Multi-Scale Windows:**
   - Window sizes: [5, 10, 15] utterances
   - Strides: [3, 5, 7]
   - Combine all for 3x more data

2. **Sequential SMOTE:**
   - Oversample CRITICAL sequences
   - Interpolate in embedding space
   - Maintain temporal coherence

3. **Back-Translation (optional):**
   - Translate to another language and back
   - Creates linguistic variation

**Commands:**
```bash
python scripts/augment_sequences.py
python -m experiments.006_augmented.run
```

---

### Experiment 007: Change Point Detection
**Status:** Pending
**Type:** Exploratory
**Effort:** HIGH (6-8 hours)

**Rationale:**
- Focus on WHEN anomaly starts, not just classification
- More aligned with real-time monitoring use case

**Approach:**
1. Use BERT+LSTM embeddings
2. Detect distribution shift over time
3. Predict: "normal → anomaly transition point"

**Evaluation:**
- Mean Absolute Error in time prediction (seconds/minutes)
- Early detection rate: % detected before X minutes

---

## Phase 3: Advanced Methods (If Time Permits)

### Experiment 008: Multi-Task Learning
- Task 1: 4-class classification
- Task 2: Binary (NORMAL vs ANOMALY)
- Task 3: Severity regression

### Experiment 009: Attention Visualization
- Analyze what utterances model focuses on
- Validate if attention matches human intuition

### Experiment 010: Domain Adaptation
- Pre-train on larger aviation corpus
- Fine-tune on CVR data

---

## Quick Start Commands (Office Computer)

### 1. Setup Environment
```bash
# Clone repo (if not already)
cd D:/projects/aviation-anomaly  # or your preferred path
git pull origin main

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 2. Run Next Experiment
```bash
# Experiment 003: Ensemble (quickest win)
python -m experiments.003_ensemble.run

# OR Experiment 004: Hierarchical Transformer
python -m experiments.004_hierarchical.run
```

### 3. Monitor Training
```bash
# Check tensorboard or logs
tail -f logs/00*/training.log

# Check GPU usage
nvidia-smi -l 1
```

---

## Experiment Template Checklist

When creating new experiment folder:

```bash
experiments/XXX_experiment_name/
├── config.yaml          # Hyperparameters
├── run.py              # Main training script
├── README.md           # Documentation
└── ERROR_ANALYSIS.md   # Post-experiment analysis
```

**Config must include:**
- Experiment ID and title
- Data configuration (source, splits)
- Model architecture
- Training hyperparameters
- Expected results

**README must include:**
- Purpose and hypothesis
- Architecture diagram
- Usage instructions
- Expected vs actual results

---

## Success Criteria

### Minimum Viable Paper
- [x] Baseline established (Exp 001)
- [x] Sequential model working (Exp 002)
- [ ] At least 3 model variants compared
- [ ] Statistical significance testing
- [ ] Ablation study

### Target Venues (Priority Order)
1. ACL Workshop on NLP for Aviation
2. Safety Science journal
3. EMNLP
4. EAAI

### Target Metrics
- **Overall Accuracy:** > 80%
- **Macro F1:** > 0.70
- **CRITICAL Recall:** > 60% (safety-critical)
- **EARLY_WARNING Recall:** > 70% (early intervention)

---

## Current Status

**Last commit:** `85a9387` - docs: add error analysis for experiment 002

**Models Available:**
- `models/001/best_model.pt` (418 MB) - Baseline BERT
- `models/002/best_model.pt` (453 MB) - BERT+LSTM

**Data Ready:**
- `data/processed/cvr_labeled.csv` (21,482 utterances)
- Total sequences: 4,190 (window=10, stride=5)

---

## Next Session Action Items

1. Run **Experiment 003** (Ensemble) - quickest win
2. If GPU available, run **Experiment 004** (Hierarchical)
3. Document results in `RESEARCH_LOG.md`
4. Commit and push after each experiment
5. Upload models to Google Drive

---

**Last Updated:** 2026-01-08
**Location:** D:/document/aviation-anomaly
**Git Branch:** main
