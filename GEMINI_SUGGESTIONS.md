# Gemini Advisory Report: Research Completion Roadmap

**Generated:** 2026-02-05
**Author:** Gemini (AI Assistant)
**Status:** ⚠️ CRITICAL ACTION REQUIRED
**Target Audience:** Future AI Agents / Developers

---

## 🛑 Critical Assessment
The current research paper (**PAPER_DRAFT_v1.md**) is **NOT READY** for publication. The core claims regarding safety-critical anomaly detection rely on experiments that are currently invalid or failed.

### Key Issues Identified
1.  **Experiment 006 (SMOTE-Augmented) is INVALID:**
    *   **Symptom:** Reports 100% Accuracy but 0% Critical Recall.
    *   **Root Cause:** The `CVRSequenceDataset` implementation in `experiments/006_smote_augmented/run.py` incorrectly processes data. It takes the *first 20 lines* of a flight case (which are almost always NORMAL) instead of applying a *sliding window* across the entire flight.
    *   **Result:** The model never sees CRITICAL examples during training or testing.
2.  **Experiment 007 (Cost-Sensitive Cascade) FAILED:**
    *   **Status:** Stage 2 model failed to converge (0.0% Critical Recall).
    *   **Analysis:** The loss function likely plateaued due to extreme cost penalties (20x) or optimization issues.

---

## 🚀 Recommended Roadmap (Step-by-Step)

<gemini_suggestion_start>

### Phase 1: Fix Experiment 006 (Highest Priority) ✅ COMPLETED
This is the lowest-hanging fruit to potentially achieve the >70% Critical Recall target.

1.  **Refactor Dataset Logic:** ✅ DONE (2026-02-05)
    *   **Source:** Copy the `create_sequences_from_df` logic from `experiments/002_bert_lstm/run.py`.
    *   **Destination:** Replace the flawed dataset construction in `experiments/006_smote_augmented/run.py`.
    *   **Goal:** Ensure sliding windows are generated so the model sees the *end* of flights (Critical/Elevated phases).
    *   **Changes Made:**
        - Replaced `CVRSequenceDataset` with `SequentialCVRDataset` using sliding window logic
        - Now uses `create_sequences_from_df()` from Exp 002
        - Label now taken from LAST utterance in window (not majority voting)
2.  **Retrain:** ⏳ READY TO RUN - Run the experiment again from scratch.
3.  **Validate:** ⏳ PENDING - Ensure `test_df` in evaluation contains all 4 classes.

### Phase 2: Execute Ablation Study (Exp 008)
To strengthen the methodology section of the paper.

1.  **Run:** `experiments/008_ablation_window_size/run.py`.
2.  **Outcome:** Produce a chart/table justifying the choice of `window_size=10`. This is standard rigor for NLP papers.

### Phase 3: Finalize Paper Strategy
Decide the narrative based on Phase 1 results:

*   **Scenario A (Exp 006 Success > 70% Recall):**
    *   **Narrative:** "Sequential SMOTE effectively solves the class imbalance problem in safety-critical NLP."
    *   **Action:** Replace Exp 003 results with Exp 006 in the Abstract and Conclusion.
*   **Scenario B (Exp 006 Fails):**
    *   **Narrative:** "Ensemble of Sequential and Static models (Exp 003) provides the most robust safety net."
    *   **Action:** Stick to Exp 003 as the champion model. Drop the strong claims about "handling imbalance" if they aren't supported by results.

<gemini_suggestion_end>

---

## Technical Context for Future Agents

*   **Repo Structure:**
    *   `experiments/002_bert_lstm/run.py`: Contains the **CORRECT** `create_sequences_from_df` sliding window logic.
    *   `experiments/006_smote_augmented/run.py`: Contains the **FLAWED** `CVRSequenceDataset` class.
*   **Data Path:** `data/cvr_labeled.csv` (requires sliding window processing to be useful).
*   **Current Best Model:** Experiment 003 (Ensemble).

---

## Status Update Log

| Date | Update | Status |
|------|--------|--------|
| 2026-02-05 | Fixed Experiment 006 sliding window bug | ✅ Completed |
| 2026-02-05 | Updated GEMINI_SUGGESTIONS.md with progress | ✅ Completed |

### Next Actions Required
1. **Run Experiment 006** - Execute `python run.py` in `experiments/006_smote_augmented/`
2. **Compare Results** - Check if CRITICAL recall improved vs Exp 003 (47% baseline)
3. **Update Paper** - If successful (>70% CRITICAL recall), update paper claims

---
*End of Advisory Report*
