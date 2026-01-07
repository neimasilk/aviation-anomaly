# Research Proposal

## Temporal Dynamics of Pilot Communication Before Aviation Accidents: A Sequence-Based Anomaly Detection Approach Using Transformer Models

---

**Researcher:** Mukhlis Amien  
**Affiliation:** STIKI Malang  
**Date:** January 2026  
**Type:** Independent Exploratory Research

---

## 1. Executive Summary

Penelitian ini mengusulkan pendekatan baru untuk mendeteksi anomali dalam komunikasi pilot sebelum kecelakaan penerbangan menggunakan analisis temporal/sekuensial berbasis Transformer. Berbeda dengan penelitian sebelumnya yang fokus pada klasifikasi statis per-utterance, penelitian ini akan mengeksplorasi **kapan dan bagaimana** pola komunikasi berubah dari normal menjadi anomali dalam konteks temporal.

**Novelty utama:** Sequential anomaly detection yang memodelkan transisi komunikasi pilot, bukan hanya klasifikasi snapshot.

---

## 2. Background & Motivation

### 2.1 Konteks Masalah

Kecelakaan penerbangan meskipun jarang, memiliki dampak fatal. Cockpit Voice Recorder (CVR) merekam komunikasi pilot yang berpotensi mengungkap early warning signs sebelum kecelakaan terjadi. Namun, analisis CVR saat ini masih bersifat post-hoc dan manual.

### 2.2 Gap yang Diidentifikasi

| Aspek | Status Riset Saat Ini | Gap |
|-------|----------------------|-----|
| Anomaly detection pada flight data (FDR) | Mature (LSTM, Autoencoder, MKAD) | - |
| NLP untuk safety reports (ASRS) | Ada beberapa paper | - |
| Sentiment analysis CVR transcripts | Baru ada (2024-2025), akurasi ~80% | Static classification only |
| **Temporal/Sequential analysis CVR** | **Belum ada** | **← TARGET PENELITIAN INI** |
| Real-time anomaly detection komunikasi | Belum ada | Future extension |

### 2.3 Mengapa Sequential Analysis?

Paper existing (BERT/RoBERTa untuk CVR) melakukan klasifikasi per-utterance secara independen. Ini mengabaikan fakta bahwa:

1. Anomali tidak muncul tiba-tiba — ada **gradasi/transisi**
2. Konteks percakapan sebelumnya mempengaruhi interpretasi
3. Timing kapan anomali mulai muncul adalah informasi kritis untuk early warning

---

## 3. Literature Review

### 3.1 Anomaly Detection in Aviation

**Flight Data Recorder (FDR) Based:**

- Das et al. (2010) - Multiple Kernel Anomaly Detection (MKAD) menggunakan One-Class SVM untuk mendeteksi anomali pada data penerbangan heterogen
- IEEE 2016 - RNN dengan LSTM/GRU untuk anomaly detection pada FDR data, mengatasi limitasi MKAD dalam mendeteksi short-term anomalies
- MDPI Aerospace 2020 - Convolutional Variational Auto-Encoder untuk unsupervised anomaly detection pada flight data

**Temuan kunci:** Semua fokus pada numerical flight parameters, bukan komunikasi verbal.

### 3.2 NLP in Aviation Safety

**Safety Report Classification:**

- BERT for Aviation Text Classification (2023) - Menggunakan ASRS (Aviation Safety Reporting System) reports untuk multi-label classification
- Aviation-BERT - Domain-specific pre-training untuk aerospace requirements classification

**Temuan kunci:** Fokus pada written reports, bukan real-time cockpit communication.

### 3.3 CVR Transcript Analysis

**Dataset Fundamental:**

- **Noort, Reader, & Gillespie (2021)** - Dataset CVR transcripts dari 172 kecelakaan penerbangan (1962-2018), 21,626 baris transkrip, open access
- Fokus penelitian mereka: Safety voice dan safety listening, bukan NLP/ML

**Sentiment/Stress Analysis pada CVR:**

- "Performance Comparison of BERT, ALBERT and RoBERTa for Sentiment Analysis in Critical Pilot Communication" (2024-2025)
  - Dataset: CVR transcripts
  - Method: Static classification per utterance
  - Results: ~80% accuracy, RoBERTa best performer
  - **Limitation: Tidak mempertimbangkan temporal dynamics**

### 3.4 Speech Under Stress Research

- Kuroda et al. (1976) - Vibration Space Shift Rate (VSSR) untuk mengukur stress pilot dari voice analysis
- Hansen & Patil (2007) - Comprehensive review speech under stress analysis
- Van Puyvelde et al. (2018) - Voice Stress Analysis framework, F0 increase sebagai indikator stress

**Temuan kunci:** Banyak riset pada acoustic features, tapi sedikit yang combine dengan NLP pada text transcripts.

### 3.5 Research Gap Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    EXISTING RESEARCH                        │
├─────────────────────────────────────────────────────────────┤
│  FDR Anomaly Detection    →  Numerical data only            │
│  ASRS Text Classification →  Written reports, not real-time │
│  CVR Sentiment Analysis   →  Static per-utterance           │
│  Voice Stress Analysis    →  Acoustic features only         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    GAP (THIS RESEARCH)                      │
├─────────────────────────────────────────────────────────────┤
│  Sequential/Temporal NLP analysis of CVR transcripts        │
│  → When does anomaly START?                                 │
│  → How does communication TRANSITION from normal to crisis? │
│  → Can we detect EARLY WARNING patterns?                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Research Questions

**RQ1:** Pada titik waktu mana (relatif terhadap kecelakaan) pola komunikasi pilot mulai menunjukkan anomali yang dapat dideteksi secara otomatis?

**RQ2:** Apakah model sequential (LSTM, Transformer dengan attention ke previous utterances) memberikan performa lebih baik dibanding static classification dalam mendeteksi transisi normal→emergency?

**RQ3:** Feature linguistik apa yang paling prediktif dalam konteks temporal untuk mendeteksi early warning signs?

**RQ4:** Bagaimana performa model pada different time windows sebelum kecelakaan (30 detik, 1 menit, 5 menit, 10 menit)?

---

## 5. Proposed Methodology

### 5.1 Dataset

**Primary Dataset:**
- Noort et al. (2021) CVR Transcript Dataset
- 172 unique transcripts
- 21,626 lines of dialogue
- Open access via ScienceDirect/Mendeley

**Supplementary Data (untuk kelas "normal"):**
- LiveATC.net archives (ATC communications)
- Simulated normal flight communications (jika tersedia)

### 5.2 Data Preprocessing

```
Raw CVR Transcript
       │
       ▼
┌──────────────────────┐
│ 1. Text Cleaning     │ → Remove timestamps, speaker tags normalization
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ 2. Temporal Ordering │ → Ensure chronological sequence
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ 3. Window Segmentation│ → Create sliding windows (e.g., 10 utterances)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ 4. Labeling          │ → Time-based labels (T-30s, T-1min, T-5min, etc.)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ 5. Feature Extraction│ → BERT embeddings + linguistic features
└──────────────────────┘
```

### 5.3 Labeling Strategy

Karena semua data dari CVR adalah kecelakaan, labeling berdasarkan temporal distance:

| Label | Definition | Hypothesis |
|-------|------------|------------|
| NORMAL | > 10 menit sebelum crash | Komunikasi rutin |
| EARLY_WARNING | 5-10 menit sebelum | Mungkin ada subtle changes |
| ELEVATED | 1-5 menit sebelum | Stress indicators emerging |
| CRITICAL | < 1 menit sebelum | Clear anomaly patterns |

### 5.4 Model Architecture

**Baseline Models:**
1. Static BERT/RoBERTa (per-utterance) — untuk comparison
2. TF-IDF + Traditional ML (SVM, Random Forest)

**Proposed Sequential Models:**

**Model A: BERT + LSTM**
```
Utterance Sequence [u1, u2, ..., un]
           │
           ▼
    ┌─────────────┐
    │ BERT Encoder│ → Per-utterance embeddings
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Bi-LSTM     │ → Capture sequential dependencies
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Attention   │ → Focus on critical utterances
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Classifier  │ → Temporal label prediction
    └─────────────┘
```

**Model B: Hierarchical Transformer**
```
Utterance Sequence [u1, u2, ..., un]
           │
           ▼
    ┌──────────────────┐
    │ Token-level      │ → BERT for each utterance
    │ Transformer      │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ Utterance-level  │ → Cross-utterance attention
    │ Transformer      │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ Sequence Label   │ → Anomaly state classification
    └──────────────────┘
```

**Model C: Change Point Detection**
```
Utterance Embeddings [e1, e2, ..., en]
           │
           ▼
    ┌──────────────────┐
    │ Sliding Window   │ → Compare adjacent windows
    │ Comparison       │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ Change Point     │ → Detect where distribution shifts
    │ Detection        │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ Anomaly Onset    │ → Identify transition point
    │ Identification   │
    └──────────────────┘
```

### 5.5 Feature Engineering

**Linguistic Features (per utterance):**
- Utterance length (word count)
- Speech rate proxy (words per estimated time)
- Sentence completeness
- Question frequency
- Exclamation/urgency markers
- Aviation-specific keyword density (mayday, emergency, terrain, etc.)
- Repetition patterns
- Negation frequency

**Sequential Features (across window):**
- Utterance length variance
- Turn-taking patterns (who speaks more?)
- Topic coherence (semantic similarity between consecutive utterances)
- Escalation patterns in urgency keywords

### 5.6 Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness |
| Macro F1-Score | Balanced performance across classes |
| Recall per class | Especially for CRITICAL class |
| Precision per class | Avoid false alarms |
| Time-to-Detection | How early can we detect? |
| AUC-ROC | Discrimination ability |

**Special Metric: Early Detection Score**
```
EDS = Σ (correct_prediction × time_before_crash) / total_predictions
```
Higher score = earlier correct detection = better model.

---

## 6. Research Roadmap

### Phase 1: Foundation (Bulan 1-2)

**Week 1-2: Dataset Acquisition & Exploration**
- [ ] Download Noort et al. dataset
- [ ] Exploratory data analysis
- [ ] Understand data structure and variables
- [ ] Identify data quality issues

**Week 3-4: Literature Deep Dive**
- [ ] Read and annotate 20 most relevant papers
- [ ] Identify exact baselines to compare
- [ ] Refine research questions if needed

**Week 5-6: Data Preprocessing Pipeline**
- [ ] Build cleaning pipeline
- [ ] Implement temporal labeling
- [ ] Create train/validation/test splits
- [ ] Handle class imbalance strategy

**Week 7-8: Baseline Implementation**
- [ ] Implement static BERT classifier (reproduce existing work)
- [ ] Implement traditional ML baselines
- [ ] Establish baseline metrics

**Deliverable:** Preprocessed dataset + baseline results

### Phase 2: Core Development (Bulan 3-4)

**Week 9-10: Sequential Model A (BERT+LSTM)**
- [ ] Implement architecture
- [ ] Hyperparameter tuning
- [ ] Initial evaluation

**Week 11-12: Sequential Model B (Hierarchical Transformer)**
- [ ] Implement architecture
- [ ] Hyperparameter tuning
- [ ] Comparative evaluation

**Week 13-14: Change Point Detection (Model C)**
- [ ] Implement architecture
- [ ] Evaluate on detecting transition points
- [ ] Compare with other models

**Week 15-16: Ablation Studies**
- [ ] Feature importance analysis
- [ ] Window size experiments
- [ ] Embedding comparison (BERT vs RoBERTa vs domain-specific)

**Deliverable:** Trained models + comparative analysis

### Phase 3: Analysis & Writing (Bulan 5-6)

**Week 17-18: Deep Analysis**
- [ ] Error analysis
- [ ] Case studies of interesting predictions
- [ ] Linguistic pattern extraction
- [ ] Visualization of results

**Week 19-20: Paper Writing**
- [ ] Draft introduction and related work
- [ ] Write methodology section
- [ ] Present results and discussion

**Week 21-22: Refinement**
- [ ] Internal review and revision
- [ ] Prepare supplementary materials
- [ ] Code and data documentation

**Week 23-24: Submission Preparation**
- [ ] Format for target venue
- [ ] Final proofreading
- [ ] Submit

**Deliverable:** Completed paper ready for submission

---

## 7. Expected Contributions

### 7.1 Scientific Contributions

1. **Novel Task Formulation:** Framing CVR analysis as temporal/sequential anomaly detection rather than static classification

2. **Benchmark Results:** Comprehensive comparison of sequential vs static models on CVR data

3. **Linguistic Insights:** Identification of temporal linguistic markers that precede aviation accidents

4. **Early Warning Framework:** Methodology for detecting communication anomalies before they escalate

### 7.2 Practical Contributions

1. **Foundation for Real-time Systems:** Results can inform design of real-time cockpit monitoring systems

2. **Training Implications:** Insights for Crew Resource Management (CRM) training programs

3. **Open Source:** Code and processed data (where legally permissible) for reproducibility

---

## 8. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dataset too small for deep learning | Medium | High | Use transfer learning, data augmentation, focus on simpler models |
| Class imbalance severe | High | Medium | SMOTE, class weights, focal loss |
| Temporal labels subjective | Medium | Medium | Multiple labeling schemes, sensitivity analysis |
| Existing paper too similar | Low | High | Focus on sequential aspect which is clearly novel |
| Computational resources limited | Medium | Low | Use Google Colab Pro, focus on efficient models |
| Results not significantly better than baseline | Medium | Medium | Contribute linguistic analysis even if ML gains modest |

---

## 9. Target Venues

**Primary Targets:**

| Venue | Type | Deadline (Typical) | Fit |
|-------|------|-------------------|-----|
| ACL Workshop on NLP for Aviation | Workshop | ~Feb/May | High |
| EMNLP | Conference | ~May | Medium |
| Safety Science | Journal | Rolling | High |
| EAAI (Engineering Applications of AI) | Journal | Rolling | Medium |

**Backup Targets:**

| Venue | Type | Notes |
|-------|------|-------|
| AACL-IJCNLP | Conference | Asia-Pacific focus |
| Journal of Safety Research | Journal | Safety-focused |
| IEEE Access | Journal | Faster review |
| arXiv | Preprint | For visibility while targeting journals |

---

## 10. Resource Requirements

### 10.1 Computational

- GPU access (Google Colab Pro sufficient for initial experiments)
- Storage for datasets and models (~10GB)

### 10.2 Software

- Python 3.8+
- PyTorch / Hugging Face Transformers
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn for visualization

### 10.3 Data

- Noort et al. dataset (free, open access)
- Optional: LiveATC recordings for normal class

### 10.4 Time

- Estimated: 6 months part-time
- Can be compressed to 4 months with dedicated effort

---

## 11. Alternative Research Directions

Jika arah utama menghadapi hambatan signifikan, berikut alternatif yang masih dalam scope:

### Alternative A: Cross-lingual CVR Analysis
- Fokus pada CVR dari berbagai bahasa
- Analisis apakah stress patterns universal atau culture-specific
- Leverage multilingual BERT

### Alternative B: Multimodal Approach (jika audio tersedia)
- Combine text features dengan acoustic features
- Pitch, tempo, volume dari audio
- More challenging but higher potential impact

### Alternative C: Synthetic Data Generation
- Train model pada simulated emergency communications
- Use LLM untuk generate synthetic training data
- Validate pada real CVR data

### Alternative D: Survey/Review Paper
- Comprehensive survey on NLP for aviation safety
- Lower novelty tapi higher acceptance rate
- Good fallback jika experimental results tidak signifikan

---

## 12. Conclusion

Penelitian ini mengusulkan pendekatan sequential/temporal untuk analisis anomali pada komunikasi pilot, mengisi gap yang jelas dalam literatur yang saat ini hanya menggunakan static classification. Dengan memanfaatkan dataset yang sudah tersedia (Noort et al., 2021) dan teknik NLP modern (Transformer-based models), penelitian ini berpotensi memberikan kontribusi signifikan pada bidang aviation safety dan computational linguistics.

**Key Differentiators:**
1. Sequential vs Static analysis
2. Temporal labeling scheme
3. Early detection focus
4. Comprehensive model comparison

**Success Criteria:**
- Demonstrate that sequential models outperform static models
- OR provide valuable linguistic insights about communication patterns before accidents
- Publishable paper at workshop/conference/journal level

---

## References

1. Noort, M. C., Reader, T. W., & Gillespie, A. (2021). Cockpit voice recorder transcript data: Capturing safety voice and safety listening during historic aviation accidents. *Data in Brief*, 39, 107602.

2. Noort, M. C., Reader, T. W., & Gillespie, A. (2021). Safety voice and safety listening during aviation accidents: Cockpit voice recordings reveal that speaking-up to power is not enough. *Safety Science*, 139, 105260.

3. Das, S., Matthews, B. L., Srivastava, A. N., & Oza, N. C. (2010). Multiple kernel learning for heterogeneous anomaly detection. *KDD*.

4. Hansen, J. H., & Patil, S. (2007). Speech under stress: Analysis, modeling and recognition. In *Speaker Classification I* (pp. 108-137). Springer.

5. Van Puyvelde, M., Neyt, X., McGlone, F., & Pattyn, N. (2018). Voice stress analysis: A new framework for voice and effort in human performance. *Frontiers in Psychology*, 9, 1994.

6. Kuroda, I., Fujiwara, O., Okamura, N., & Utsuki, N. (1976). Method for determining pilot stress through analysis of voice communication. *Aviation, Space, and Environmental Medicine*, 47(5), 528-533.

7. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

8. Liu, Y., Ott, M., Goyal, N., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

---

## Appendix A: Dataset Schema (Noort et al.)

```
Variables:
- case_id: Unique identifier for each accident
- date: Date of accident
- location: Location of accident
- cvr_message: The spoken text
- cvr_speaker_role: Captain, First Officer, Flight Engineer, etc.
- cvr_turn_number: Sequential turn number
- coding_hazard_raised: Whether hazard was mentioned
- coding_listening_response: How others responded
- ...
```

## Appendix B: Potential Feature List

**Lexical Features:**
- Word count
- Unique word ratio
- Aviation jargon frequency
- Emergency keyword presence (mayday, terrain, pull up, etc.)

**Syntactic Features:**
- Sentence completeness
- Question ratio
- Imperative ratio
- Fragment ratio

**Discourse Features:**
- Turn length ratio (captain vs first officer)
- Interruption patterns
- Repetition frequency
- Topic continuity

**Temporal Features:**
- Speaking rate changes
- Silence/pause patterns (if available)
- Response latency (estimated)

---

*End of Proposal*
