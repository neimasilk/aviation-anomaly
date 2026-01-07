# Aviation Anomaly Detection

**Temporal Dynamics of Pilot Communication Before Aviation Accidents: A Sequence-Based Anomaly Detection Approach Using Transformer Models**

Research by Mukhlis Amien (STIKI Malang, 2026)

## Overview

This project investigates temporal and sequential patterns in pilot communication from Cockpit Voice Recorder (CVR) transcripts to detect early warning signs before aviation accidents. Unlike existing approaches that use static per-utterance classification, this research models how communication patterns transition from normal to anomalous over time.

### Key Features

- **Sequential Analysis**: Models temporal dependencies in communication patterns
- **Multi-Stage Detection**: Labels based on time-before-crash (NORMAL, EARLY_WARNING, ELEVATED, CRITICAL)
- **Multiple Architectures**: BERT+LSTM, Hierarchical Transformer, Change Point Detection
- **Data Augmentation**: DeepSeek API integration for synthetic data generation

## Project Structure

```
aviation-anomaly/
├── config/
│   ├── default.yaml      # Default configuration
│   └── .env.example      # Example environment variables
├── data/
│   ├── raw/              # Raw datasets (not in git)
│   ├── processed/        # Processed data (not in git)
│   └── .gitkeep
├── src/
│   ├── data/             # Data preprocessing
│   ├── models/           # Model architectures
│   ├── train/            # Training utilities
│   ├── evaluate/         # Evaluation scripts
│   ├── utils/            # Utilities (config, DeepSeek API)
│   └── scripts/          # Entry point scripts
├── models/               # Trained models (not in git)
├── logs/                 # Training logs (not in git)
├── notebooks/            # Jupyter notebooks
├── experiments/          # Experiment outputs (not in git)
├── tests/                # Unit tests
├── .env                  # API keys (NOT in git)
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── CLAUDE.md            # Claude Code guidance
```

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd aviation-anomaly
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your DeepSeek API key
```

## Usage

### Data Preprocessing

```bash
python -m src.scripts.preprocess --input data/raw/cvr_transcripts.csv
```

### Training Models

```bash
# Train BERT+LSTM model
python -m src.scripts.train --model bert_lstm --epochs 50

# Train Hierarchical Transformer
python -m src.scripts.train --model hierarchical_transformer

# Train on GPU machine
python -m src.scripts.train --model bert_lstm --device cuda
```

### Evaluation

```bash
python -m src.scripts.evaluate --checkpoint models/checkpoints/best_model.pt
```

## Multi-Computer Workflow

This project is designed to work across multiple computers:

1. **Regular Computer**: Development, preprocessing, small-scale experiments
2. **Training Computer**: GPU training, large-scale experiments

### Workflow

```bash
# On regular computer - make changes and push
git add .
git commit -m "description"
git push

# On training computer - pull latest changes
git pull

# Run training
python -m src.scripts.train --model bert_lstm

# Commit and push results (config, logs, not data/models)
git add config/ src/ notebooks/
git commit -m "update training config"
git push
```

### What Gets Tracked

- **Tracked**: Source code, config, notebooks, documentation
- **NOT Tracked**: Data files, models, logs, API keys (see .gitignore)

## Dataset

The primary dataset is from:

> Noort, M. C., Reader, T. W., & Gillespie, A. (2021). Cockpit voice recorder transcript data: Capturing safety voice and safety listening during historic aviation accidents. *Data in Brief*, 39, 107602.

- 172 unique transcripts (1962-2018)
- 21,626 lines of dialogue
- Available at: https://doi.org/10.1016/j.dib.2021.107602

## Temporal Labeling

Since all CVR data comes from accidents, labels are based on time-before-crash:

| Label | Time Before Crash | Description |
|-------|-------------------|-------------|
| NORMAL | > 10 minutes | Routine communication |
| EARLY_WARNING | 5-10 minutes | Subtle changes emerging |
| ELEVATED | 1-5 minutes | Stress indicators visible |
| CRITICAL | < 1 minute | Clear anomaly patterns |

## Model Architectures

### Model A: BERT + LSTM
- Per-utterance BERT encoding
- Bi-LSTM for sequential dependencies
- Attention mechanism
- Classification head

### Model B: Hierarchical Transformer
- Token-level Transformer (BERT)
- Utterance-level Transformer
- Cross-utterance attention

### Model C: Change Point Detection
- Sliding window comparison
- Distribution shift detection
- Anomaly onset identification

## Evaluation Metrics

- Standard: Accuracy, Macro F1-Score, AUC-ROC
- Per-class: Precision, Recall, F1
- Custom: **Early Detection Score (EDS)**
  ```
  EDS = Σ (correct_prediction × time_before_crash) / total_predictions
  ```

## Research Roadmap

- **Phase 1 (Months 1-2)**: Dataset acquisition, preprocessing, baselines
- **Phase 2 (Months 3-4)**: Core model development, ablation studies
- **Phase 3 (Months 5-6)**: Analysis, paper writing, submission

## License

MIT License

## Citation

```bibtex
@misc{amien2026aviation,
  title={Temporal Dynamics of Pilot Communication Before Aviation Accidents: A Sequence-Based Anomaly Detection Approach Using Transformer Models},
  author={Amien, Mukhlis},
  year={2026},
  institution={STIKI Malang}
}
```
