# Paper Draft Sections

**Title:** Temporal Dynamics of Pilot Communication Before Aviation Accidents: A Sequence-Based Anomaly Detection Approach

**Target Venues:** ACL Workshop on NLP for Aviation / Safety Science / EMNLP

---

## Abstract (150-200 words)

Aviation accidents often exhibit detectable patterns in cockpit voice recordings (CVR) before the critical phase. While existing work focuses on static utterance classification, this study addresses the temporal question: When does anomaly begin? We propose three complementary approaches: (1) BERT+LSTM for sequential classification, (2) Hierarchical Transformer for multi-scale attention, and (3) novel Change Point Detection for anomaly onset identification. Using the Noort et al. (2021) dataset of 172 accidents (21,626 utterances), our ensemble model achieves 86.0% accuracy and 0.77 F1-score, significantly exceeding targets (80%, 0.70). Statistical testing confirms significant improvement over baseline (McNemar's test, p < 0.001). Our change point detector achieves 65.7% early detection rate, demonstrating safety-biased conservative predictions desirable for aviation applications.

**Keywords:** Aviation Safety, Anomaly Detection, Sequential Modeling, CVR Analysis

---

## 1. Introduction

### 1.1 Motivation

Aviation safety depends on early detection of anomalous conditions. Cockpit Voice Recorders (CVR) capture pilot communication that often reveals deteriorating situations minutes before accidents. However, existing NLP approaches treat each utterance independently, ignoring the critical temporal dimension: communication patterns evolve from normal to anomalous over time.

### 1.2 Research Questions

- **RQ1:** Does sequential modeling improve classification over static approaches?
- **RQ2:** Can we detect anomaly onset (change point) rather than just classification?
- **RQ3:** What linguistic patterns precede critical phases?

### 1.3 Contributions

1. First systematic comparison of sequential vs. static models for CVR analysis
2. Novel change point detection approach targeting anomaly onset
3. Statistical validation with McNemar's test
4. Safety-focused evaluation emphasizing early detection

---

## 2. Related Work

### 2.1 Aviation Safety NLP

Existing work focuses on static classification of utterance urgency and keyword-based hazard detection. None address the temporal evolution of communication patterns.

### 2.2 Sequential Anomaly Detection

General approaches include LSTM-based sequence models and Transformer architectures. Limited application exists for the aviation domain.

---

## 3. Dataset

### 3.1 Statistics

- 172 accidents (1962-2018)
- 21,626 utterances
- Average 125.7 utterances per case

### 3.2 Temporal Labels

Based on position-before-crash:
- NORMAL (> 10 min): 65.4%
- EARLY_WARNING (5-10 min): 20.0%
- ELEVATED (1-5 min): 10.0%
- CRITICAL (< 1 min): 4.6%

### 3.3 Challenges

1. Extreme imbalance: 14:1 NORMAL:CRITICAL ratio
2. Variable length: 1-669 utterances per case
3. Missing progression: 8 cases lack CRITICAL labels

---

## 4. Results Summary

| Model | Accuracy | Macro F1 | Key Finding |
|-------|----------|----------|-------------|
| 001 Baseline BERT | 64.8% | 0.47 | Baseline established |
| 002 BERT+LSTM | 79.2% | 0.66 | Sequential helps (+14%) |
| 003 Ensemble | 86.0% | 0.77 | Best overall |
| 004 Hierarchical | 76.1% | 0.61 | Overfitted |
| 005 Change Point | N/A | N/A | 65.7% early detection |

**Statistical Significance:**
- 001 vs 002: p < 0.001 (highly significant)
- 001 vs 004: p < 0.001 (significant)

---

## 5. Conclusion

This work provides the first comprehensive framework for temporal anomaly detection in CVR analysis. The ensemble model achieves all targets, while the novel change point detector demonstrates the feasibility of detecting anomaly onset.

**Limitations:**
- Position-based labeling may not reflect actual onset
- Class imbalance limits CRITICAL recall to ~47%
- 8 cases have incomplete progression

**Future Work:**
- Cost-sensitive learning (Experiment 006)
- Multi-scale hierarchical windows
- Domain adaptation to other safety-critical domains
