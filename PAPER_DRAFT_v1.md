# Beyond Static Sentiments: Sequential Anomaly Detection in Cockpit Voice Recorder Transcripts using Deep Learning

**Author:** Mukhlis Amien  
**Affiliation:** STIKI Malang  
**Date:** January 2026

---

## Abstract

**Context:** Cockpit Voice Recorder (CVR) analysis is a critical component of aviation accident investigation. While numerical Flight Data Recorder (FDR) anomaly detection is well-established, automated analysis of CVR transcripts remains largely manual or limited to static, utterance-level sentiment classification.
**Problem:** Existing Natural Language Processing (NLP) approaches treat pilot communications as isolated sentences, ignoring the temporal dynamics and sequential context that characterize the transition from normal flight to emergency situations.
**Method:** This study proposes a sequential anomaly detection framework that models the temporal dependencies in pilot discourse. We utilize the Noort et al. (2021) dataset comprising 172 accident transcripts. We compare a static Baseline (BERT) against sequential architectures (BERT-LSTM, Hierarchical Transformer, and Ensemble methods) using a sliding window approach with four hazard levels: Normal, Early Warning, Elevated, and Critical.
**Results:** Our sequential Ensemble model achieves a Macro F1-score of **0.77** and Accuracy of **86.0%**, significantly outperforming the static baseline (F1 0.47, Accuracy 64.8%). Crucially, the sequential approach reduces safety-critical missed detections (anomalies misclassified as normal) by **91%** compared to the baseline.
**Conclusion:** The results demonstrate that modeling the *sequence* of communication, rather than just the content of individual utterances, is essential for effective anomaly detection in aviation. This work provides a foundation for future real-time cockpit monitoring systems.

**Keywords:** Aviation Safety, NLP, Anomaly Detection, CVR Analysis, Sequential Modeling, Deep Learning.

---

## 1. Introduction

Aviation accidents are rarely the result of a single catastrophic failure; rather, they are often the culmination of a chain of events, errors, and miscommunications. The Cockpit Voice Recorder (CVR) captures the verbal interactions of the flight crew, offering unique insights into the human factors precipitating these events. Traditionally, CVR analysis is a post-hoc, manual process performed by expert investigators.

Recent advancements in Natural Language Processing (NLP) have opened avenues for automated analysis. However, prior works have predominantly focused on **static classification**—analyzing individual utterances to detect sentiment or specific keywords (e.g., "Mayday"). While useful, this approach fails to capture the *temporal dynamics* of a developing crisis. A calm statement like "turn left" has a vastly different implication during a routine approach versus immediately following a terrain warning.

This research addresses this gap by framing CVR analysis as a **Sequential Anomaly Detection** task. We hypothesize that the *pattern of transition* in communication offers stronger predictive signals than isolated words.

Our contributions are as follows:
1.  We formulate a temporal labeling strategy for CVR transcripts based on time-to-impact.
2.  We benchmark sequential models (Bi-LSTM, Hierarchical Transformers) against state-of-the-art static BERT classifiers.
3.  We demonstrate that sequential modeling improves the detection of "Early Warning" signs by over 30% in recall and reduces dangerous missed detections by 91%.

---

## 2. Methodology

### 2.1 Dataset
We utilize the dataset curated by Noort et al. (2021), containing transcripts from 172 aviation accidents (1962–2018). The raw data consists of 21,626 lines of dialogue.

### 2.2 Temporal Labeling & Windowing
Unlike previous studies that rely on annotator-perceived stress, we use an objective time-based labeling scheme relative to the accident event ($t_{end}$):
*   **NORMAL:** $t < t_{end} - 10 	ext{ min}$
*   **EARLY WARNING:** $t_{end} - 10 	ext{ min} ≤ t < t_{end} - 5 	ext{ min}$
*   **ELEVATED:** $t_{end} - 5 	ext{ min} ≤ t < t_{end} - 1 	ext{ min}$
*   **CRITICAL:** $t < t_{end} - 1 	ext{ min}$

To capture context, we employ a sliding window approach:
*   **Window Size:** 10 consecutive utterances
*   **Stride:** 5 utterances (50% overlap)
*   **Input:** Sequence of 10 text strings
*   **Output:** The label of the final utterance in the window

### 2.3 Architectures
We compare three distinct approaches:
1.  **Baseline (Static):** A pre-trained BERT model classifying single utterances independently.
2.  **Sequential (BERT-LSTM):** BERT embeddings fed into a Bidirectional LSTM to capture temporal dependencies across the window.
3.  **Ensemble:** A soft-voting mechanism combining the probability distributions of the best static and sequential models.

---

## 3. Results

### 3.1 Model Comparison

Table 1 summarizes the performance of all experimental models on the held-out test set.

| Model | Accuracy | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Baseline (Static BERT) | 64.82% | 0.4734 | 0.516 | 0.449 |
| Hierarchical Transformer | 76.13% | 0.6097 | 0.628 | 0.594 |
| BERT + Bi-LSTM | 79.17% | 0.6589 | 0.693 | 0.635 |
| **Ensemble (Ours)** | **86.04%** | **0.7668** | **0.781** | **0.755** |

*Table 1: Performance comparison. The Ensemble model significantly outperforms the static baseline.*

### 3.2 Safety-Critical Analysis

A key metric in aviation safety is the rate of **Missed Detections** (False Negatives), where a hazardous situation is misclassified as 'Normal'.

*   **Baseline Model:** 534 missed detections.
*   **Sequential Model:** 46 missed detections.

**Result:** The introduction of sequential context reduced safety-critical errors by **91.4%**.

### 3.3 Class-wise Performance

The "Early Warning" class, crucial for preventative measures, saw the largest improvement. Recall increased from 35.6% (Baseline) to 66.7% (Sequential), indicating the model effectively learns subtle precursors to accidents that static analysis misses.

---

## 4. Discussion

### 4.1 The Importance of Context
Our error analysis reveals that the static model fails on ambiguous short commands. The sequential model, however, successfully uses the history of the previous 9 utterances to disambiguate the urgency of the current command.

### 4.2 Overfitting in Complex Models
Interestingly, the Hierarchical Transformer (Exp 004) underperformed compared to the simpler BERT-LSTM (Exp 002). We attribute this to the relatively small dataset size (~4,000 sequences). The inductive bias of the LSTM appears better suited for this scale than the data-hungry Transformer attention mechanisms.

### 4.3 Limitations
The "Elevated" class (1-5 mins before impact) remains the most difficult to classify, often confused with "Early Warning". This suggests a gradual rather than discrete transition in communication patterns.

---

## 5. Conclusion

This study establishes that sequential modeling is a prerequisite for effective automated CVR analysis. By leveraging the temporal nature of pilot discourse, we can detect anomalies earlier and more reliably than traditional static methods. Future work will focus on integrating acoustic features (prosody) to further enhance detection capabilities.
