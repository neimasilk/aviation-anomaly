# Experiment 002: BERT+LSTM - Sequential Pattern Modeling

**Date:** 2026-01-07
**Status:** In Progress
**Type:** Sequential Model

## Overview

This experiment extends the baseline BERT model by incorporating sequential dependencies through a Bi-LSTM layer. The hypothesis is that modeling the temporal flow of cockpit conversations will improve detection of early warning signs.

## Key Differences from Baseline (Exp 001)

| Aspect | Baseline (001) | Sequential (002) |
|--------|---------------|------------------|
| Input | Single utterance | Sequence of 10 utterances |
| Context | None | Previous utterances |
| Architecture | BERT + Classifier | BERT + Bi-LSTM + Attention + Classifier |
| Window | Static (1) | Sliding (10 utterances, stride=5) |

## Architecture

```
Input Sequence (10 utterances)
         ↓
    BERT Encoder (per utterance)
         ↓
   [CLS] Token Embeddings
         ↓
    Bi-LSTM (256 hidden, 2 layers)
         ↓
    Attention Mechanism
         ↓
   Classification Head
         ↓
  NORMAL / EARLY_WARNING / ELEVATED / CRITICAL
```

## Configuration

- **Model:** bert-base-uncased + Bi-LSTM (256 hidden, 2 layers)
- **Window Size:** 10 utterances
- **Stride:** 5 utterances (50% overlap)
- **Batch Size:** 16 (reduced due to sequential data)
- **Learning Rate:** 2e-5
- **Max Epochs:** 15

## Expected Results

Compared to baseline:
- **Accuracy:** > 65% (baseline: 64.8%)
- **Macro F1:** > 0.50 (baseline: 0.47)

The sequential context should particularly improve:
1. EARLY_WARNING detection (less confusion with NORMAL)
2. CRITICAL detection (better temporal context)

## Usage

```bash
cd experiments/002_bert_lstm
python run.py
```

## Results

*To be filled after experiment completes.*