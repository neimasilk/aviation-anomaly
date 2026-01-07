# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for "Temporal Dynamics of Pilot Communication Before Aviation Accidents: A Sequence-Based Anomaly Detection Approach Using Transformer Models" by Mukhlis Amien (STIKI Malang, January 2026).

**Research Focus:** Sequential/temporal NLP analysis of Cockpit Voice Recorder (CVR) transcripts to detect early warning signs before aviation accidents. Unlike existing static per-utterance classification, this research models how communication patterns transition from normal to anomalous over time.

**Current Status:** Research proposal phase - no implementation yet. The repository contains only the research proposal document (`research_proposal.md`).

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

## Implementation Commands (When Code is Added)

```bash
# Environment setup
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn

# Data preprocessing
python preprocess.py

# Training different model architectures
python train.py --model bert_lstm
python train.py --model hierarchical_transformer
python train.py --model change_point

# Evaluation
python evaluate.py

# Results visualization
python visualize_results.py
```

## Research Roadmap

**Phase 1 (Months 1-2):** Dataset acquisition, preprocessing, baseline models (static BERT)

**Phase 2 (Months 3-4):** Core sequential model development, ablation studies

**Phase 3 (Months 5-6):** Analysis, paper writing, submission

## Evaluation Metrics

- Standard: Accuracy, Macro F1-Score, AUC-ROC, Recall/Precision per class
- Custom: **Early Detection Score (EDS)** - rewards earlier correct predictions

```
EDS = Σ (correct_prediction × time_before_crash) / total_predictions
```

## Target Venues

- ACL Workshop on NLP for Aviation (primary)
- Safety Science journal
- EMNLP
- EAAI (Engineering Applications of AI)

## Linguistic Features

**Per-utterance:** Word count, speech rate proxy, sentence completeness, question frequency, urgency markers, aviation keywords (mayday, emergency, terrain), repetition patterns

**Sequential (across window):** Utterance length variance, turn-taking patterns, topic coherence, escalation patterns
