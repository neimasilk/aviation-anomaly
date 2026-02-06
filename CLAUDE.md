# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for "Temporal Dynamics of Pilot Communication Before Aviation Accidents: A Sequence-Based Anomaly Detection Approach Using Transformer Models" by Mukhlis Amien (STIKI Malang, January 2026).

**Research Focus:** Sequential/temporal NLP analysis of Cockpit Voice Recorder (CVR) transcripts to detect early warning signs before aviation accidents. Unlike existing static per-utterance classification, this research models how communication patterns transition from normal to anomalous over time.

**Current Status:** Partial execution complete. 8 experiments completed (001-005), Exp 010 (Traditional Baselines) ✅ DONE, 5 new experiments ready to run (006, 008, 011, 013, 014), 2 blocked (009, 012 need API key). Targeting Safety Science journal (IF ~6.1). See `EXECUTION_GUIDE.md` for running instructions and `BABY_STEPS.md` for step-by-step guide.

**Latest Results:**
- Exp 010: 9 traditional ML baselines tested, best is TF-IDF + SVM (Acc: 76.0%, F1: 0.635)
- All statistical tests show significant improvements (p<0.001) between model variants

**New Paper Title (draft):** "From Position to Content: A Comprehensive Benchmark of Temporal Anomaly Detection Methods in Cockpit Voice Recorder Transcripts"

## Proposed Tech Stack

- **Language:** Python 3.8+
- **ML Framework:** PyTorch, Hugging Face Transformers
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Hardware:** GPU (Google Colab Pro sufficient)

## Key Research Concepts

### Temporal Labeling Strategy
All CVR data comes from accident recordings, so labels are based on time-before-crash:

| Label | Time Before Crash | Hypothesis |
|-------|-------------------|------------|
| NORMAL | > 10 minutes | Routine communication |
| EARLY_WARNING | 5-10 minutes | Subtle changes emerging |
| ELEVATED | 1-5 minutes | Stress indicators visible |
| CRITICAL | < 1 minute | Clear anomaly patterns |

### Model Architectures

**Model A (BERT + LSTM):** Per-utterance BERT embeddings → Bi-LSTM → Attention → Classifier

**Model B (Hierarchical Transformer):** Token-level Transformer → Utterance-level Transformer → Sequence Label

**Model C (Change Point Detection):** Sliding window comparison → Distribution shift detection → Anomaly onset identification

### Primary Dataset

Noort et al. (2021) CVR Transcript Dataset:
- 172 unique transcripts (1962-2018 accidents)
- 21,626 lines of dialogue
- Open access via ScienceDirect/Mendeley
- Variables: `case_id`, `cvr_message`, `cvr_speaker_role`, `cvr_turn_number`, etc.

## Key Documentation Files

- **`EXECUTION_GUIDE.md`** — Full execution plan with phases, prerequisites, and troubleshooting
- **`BABY_STEPS.md`** — Ultra-simple step-by-step for junior developers or less capable AI models
- **`IMPLEMENTATION_STATUS.md`** — What was implemented, why, and what's left
- **`CRITICAL_REVIEW.md`** — Known weaknesses and how they're addressed

## Implementation Commands

```bash
# Setup
pip install -e .
pip install xgboost sentence-transformers  # for Exp 010

# Run experiments (in priority order):
python experiments/010_traditional_baselines/run.py     # CPU only, 30 min
python experiments/011_deberta_lstm/run.py              # GPU, 2-4 hrs
python experiments/006_smote_augmented/run.py           # GPU, 2-4 hrs
python experiments/014_kfold_evaluation/run.py          # GPU, 12-20 hrs

# LLM annotation (needs API key):
python scripts/annotation/llm_annotate.py --provider deepseek

# Labeling comparison (needs annotation done first):
python experiments/009_labeling_comparison/run.py

# Analysis:
python scripts/analysis/statistical_testing.py
python scripts/analysis/attention_visualization.py
python scripts/analysis/error_analysis.py
```

## Completed Experiments

| Exp | Name | Accuracy | Macro F1 | Status |
|-----|------|----------|----------|--------|
| 001 | Baseline BERT | 64.8% | 0.473 | Done |
| 002 | BERT+LSTM | 79.2% | 0.659 | Done |
| 003 | Ensemble | **86.0%** | **0.767** | Done (BEST) |
| 004 | Hierarchical Transformer | 76.1% | 0.610 | Done |
| 005 | Change Point Detection | MAE 49.1 | - | Done |
| 006 | SMOTE-Augmented | - | - | Fixed, ready to run |
| 007 | Cost-Sensitive Cascade | - | - | Failed (Stage 2) |
| 008 | Window Size Ablation | - | - | Ready to run |

## New Experiments (Coded, Not Yet Run)

| Exp | Name | Purpose | Needs |
|-----|------|---------|-------|
| 009 | Labeling Comparison | **Main contribution #1** - 3 labeling strategies | GPU + LLM annotations |
| 010 | Traditional Baselines | 9 ML baselines (TF-IDF, SVM, XGBoost, etc.) | CPU only |
| 011 | DeBERTa-v3 + LSTM | Swap BERT → DeBERTa for performance boost | GPU |
| 012 | Few-Shot LLM | Can LLM classify without training? | API key |
| 013 | QLoRA Fine-Tune | Fine-tune Phi-3-mini with 4-bit quant | GPU 16GB |
| 014 | K-Fold CV | 5-fold stratified by case_id | GPU |

## Research Roadmap (Updated)

**Phase 0** [DONE]: Fix Exp 006, prepare Exp 008
**Phase 1** [CODE DONE]: LLM annotation + content-based labeling
**Phase 2** [CODE DONE]: Traditional ML baselines
**Phase 3** [CODE DONE]: Foundation model experiments (DeBERTa, LLM few-shot, QLoRA)
**Phase 4** [CODE DONE]: K-Fold CV + statistical testing + safety metrics
**Phase 5** [CODE DONE]: Attention visualization + error analysis
**Phase 6** [TODO]: Data augmentation (stretch goal)
**Phase 7** [TODO]: Paper writing + 40+ references

## Evaluation Metrics

- Standard: Accuracy, Macro F1-Score, AUC-ROC, Recall/Precision per class
- Custom: **Early Detection Score (EDS)** - rewards earlier correct predictions

```
EDS = Σ (correct_prediction × time_before_crash) / total_predictions
```

## Target Venues (Updated)

| Priority | Journal | IF | Fit |
|----------|---------|-----|-----|
| 1 | **Safety Science** | ~6.1 | Aviation safety NLP, perfect domain match |
| 2 | Expert Systems with Applications | ~8.0 | Benchmark/comparison papers |
| 3 | Engineering Applications of AI | ~8.0 | Real-world AI applications |

## 3 Main Contributions

1. **Content-aware labeling methodology** using LLMs vs position-based (Exp 009)
2. **Comprehensive benchmark**: traditional ML vs transformers vs LLM (Exp 010-013)
3. **Safety-aware evaluation framework**: EDS, Safety-Weighted F1 (`src/evaluate/safety_metrics.py`)

## Linguistic Features

**Per-utterance:** Word count, speech rate proxy, sentence completeness, question frequency, urgency markers, aviation keywords (mayday, emergency, terrain), repetition patterns

**Sequential (across window):** Utterance length variance, turn-taking patterns, topic coherence, escalation patterns
