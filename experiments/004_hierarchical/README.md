# Experiment 004: Hierarchical Transformer

**Status:** In Progress
**Created:** 2026-01-08
**Tags:** hierarchical, transformer, sequential, model_b

## Overview

Hierarchical Transformer architecture for sequential CVR analysis. This is **Model B** from the research proposal - an alternative to BERT+LSTM that uses Transformer layers instead of LSTM for modeling temporal dependencies between utterances.

## Architecture

### Two-Level Hierarchy

1. **Token-Level (Utterance Encoder):**
   - BERT encodes each utterance into a fixed-length embedding
   - Uses [CLS] token representation

2. **Utterance-Level (Sequence Modeler):**
   - Transformer encoder layers process the sequence of utterance embeddings
   - Multi-head self-attention captures temporal dependencies
   - Positional encoding maintains order information

3. **Classification:**
   - Global attention pooling with learnable query
   - Feed-forward network for final classification

### Hypothesis

The Hierarchical Transformer will:
1. Better capture long-range dependencies in sequences (vs LSTM)
2. Provide more interpretable attention weights
3. Match or exceed BERT+LSTM performance

**Target:** Accuracy >= 0.80, Macro F1 >= 0.70
**Optimistic:** Accuracy >= 0.86 (competitive with Ensemble)

## Model Configuration

```yaml
model:
  type: "hierarchical_transformer"
  encoder: "bert-base-uncased"
  d_model: 768
  n_heads: 8
  n_layers: 4
  dim_feedforward: 2048
  dropout: 0.1
```

## Comparison: BERT+LSTM vs Hierarchical Transformer

| Aspect | BERT+LSTM (002) | Hierarchical (004) |
|--------|-----------------|-------------------|
| Sequence Model | Bi-LSTM | Transformer Encoder |
| Temporal Reach | Limited by hidden state | Full attention over sequence |
| Parallelization | Sequential | Parallel |
| Interpretability | Attention over LSTM output | Multi-head self-attention |
| Parameters | ~110M | ~110M + positional |

## Usage

```bash
# Run from project root
cd experiments/004_hierarchical
python run.py

# Or with explicit device
python run.py --device cuda
```

## Output

Results will be saved to:
- `outputs/experiments/004/` - Metrics, comparisons, confusion matrices
- `models/004/` - Model checkpoints
- `logs/004/` - Training logs

## Expected Results

| Metric | Exp 001 | Exp 002 | Exp 003 | Exp 004 (Target) |
|--------|---------|---------|---------|------------------|
| Accuracy | 0.6482 | 0.7917 | 0.8604 | >= 0.80 |
| Macro F1 | 0.4734 | 0.6589 | 0.7668 | >= 0.70 |

**Potential advantages over Exp 002:**
- Better performance on longer sequences
- More interpretable attention patterns
- Competitive with Ensemble (003) if attention works well

## Files

- `config.yaml` - Experiment configuration
- `run.py` - Training and evaluation script
- `README.md` - This file

## References

- Base Model: `src/models/hierarchical_transformer.py`
- Exp 001: `experiments/001_baseline_bert/`
- Exp 002: `experiments/002_bert_lstm/`
- Exp 003: `experiments/003_ensemble/`
