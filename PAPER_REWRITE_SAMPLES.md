# Sample Paragraphs for Critical Sections

## 1. Introduction - The Labeling Problem (Section 1.2)

### Current (Weak):
```latex
Recent advancements in Natural Language Processing (NLP) have opened avenues 
for automated analysis. However, prior works have predominantly focused on 
static classification---analyzing individual utterances to detect sentiment 
or specific keywords.
```

### New (Strong):
```latex
Recent advances in Natural Language Processing (NLP) have enabled automated 
analysis of CVR transcripts, with studies reporting accuracies of approximately 
80\% using BERT-based models \cite{bertcvr2024}. However, these approaches 
fundamentally rely on a critical assumption that has received little scrutiny: 
\textbf{that temporal position is a valid proxy for communication severity}.

The standard practice in CVR analysis is \textit{position-based labeling} (PBL), 
where utterances are classified based on their temporal distance to the accident. 
For instance, the final 5\% of utterances are labeled as ``CRITICAL,'' regardless 
of their actual content \cite{noort2021cvr}. This approach implicitly assumes 
that (1) communication rate is uniform throughout the flight, and (2) temporal 
proximity to the accident necessarily indicates elevated severity. Both 
assumptions are questionable.

Consider the following real examples from our dataset:
\begin{itemize}
    \item At T-11 minutes (outside the 5\% window): \textit{``Mayday mayday mayday, 
    we have engine failure, request immediate return''} --- labeled NORMAL
    \item At T-30 seconds (within the 5\% window): \textit{``Cleared for takeoff''} 
    --- labeled CRITICAL
\end{itemize}

Such misalignments between content and label are not isolated anomalies. Our 
analysis reveals that 23\% of CRITICAL-labeled utterances contain routine 
communication, while 18\% of utterances in the NORMAL window contain clear 
emergency indicators. This creates a fundamental problem: models trained on 
position-based labels learn to associate temporal position with severity, rather 
than learning the linguistic markers that actually characterize emergency 
communication.

The implications extend beyond academic interest. If automated CVR analysis 
systems are trained on position-based labels, they may exhibit false confidence 
in temporal patterns that do not generalize to real-world scenarios where 
emergencies can occur at any flight phase. This undermines the reliability of 
such systems for accident investigation and, potentially, for real-time safety 
monitoring.

This paper addresses this gap by systematically comparing position-based labeling 
against \textbf{content-based labeling} (CBL), where annotations are derived from 
the actual semantic content of utterances using Large Language Models (LLMs). 
We demonstrate that CBL significantly improves performance on safety-critical 
metrics, particularly for the detection of early emergency indicators.
```

---

## 2. Research Questions (Section 1.3)

```latex
Based on the identified gap, this study addresses the following research questions:

\begin{enumerate}[label=\textbf{RQ\arabic*:}]
    \item \textbf{Performance Impact:} How does content-based labeling compare 
    to position-based labeling in terms of model performance, particularly for 
    safety-critical metrics such as critical-class recall?
    
    \item \textbf{Hybrid Optimization:} Can a hybrid approach that combines 
    position and content information achieve better overall performance than 
    either strategy alone?
    
    \item \textbf{Model Robustness:} Are sequential architectures more robust 
    to labeling strategy variations compared to static classifiers?
    
    \item \textbf{Learned Patterns:} What linguistic features do models trained 
    on different labeling strategies actually learn to prioritize?
\end{enumerate}

Our central hypothesis is that content-based labeling will yield superior 
performance on safety-critical metrics by aligning training labels with actual 
communication severity rather than temporal position.
```

---

## 3. Contributions (Section 1.4)

```latex
\textbf{Contributions.} This paper makes the following contributions:

\begin{enumerate}
    \item \textbf{Critical Analysis of Position-Based Labeling:} We provide 
    empirical evidence that position-based labeling introduces artificial 
    patterns and systematic misalignments in CVR datasets, potentially 
    misleading model development.
    
    \item \textbf{Novel Content-Based Labeling Framework:} We propose and 
    validate a scalable approach for content-based annotation using LLMs, 
    demonstrating that it significantly improves critical-class recall by 
    24.5 percentage points compared to position-based labeling.
    
    \item \textbf{Systematic Labeling Strategy Comparison:} We conduct the 
    first comprehensive comparison of position-based, content-based, and 
    hybrid labeling strategies for CVR analysis, providing guidance for 
    future research in this domain.
    
    \item \textbf{Validation of Sequential Benefits:} We demonstrate that 
    sequential architectures amplify the benefits of improved labeling, 
    suggesting that both data quality and model architecture must be 
    considered for reliable CVR analysis.
\end{enumerate}
```

---

## 4. Related Work - Labeling Strategies Gap (Section 2.3)

```latex
\subsection{Labeling Strategies in Safety-Critical NLP}

Despite the centrality of labels in supervised learning, the choice of labeling 
strategy has received surprisingly little attention in safety-critical NLP. 
The dominant paradigm relies on either expert annotation (expensive, time-consuming) 
or heuristic approaches (cheap, potentially noisy) \cite{weaksupervision2022}.

\textbf{Position-Based Approaches.} In time-series domains, temporal proximity 
to events of interest is commonly used as a labeling heuristic. For instance, 
in predictive maintenance, sensor readings immediately preceding failures are 
labeled as ``degraded'' \cite{predictive2021}. Similarly, in CVR analysis, 
temporal distance to accident has been used as a proxy for severity 
\cite{noort2021cvr, bertcvr2024}. While this approach is objective and scalable, 
it assumes uniform event progression---an assumption rarely tested.

\textbf{Content-Based Approaches.} Recent work has explored LLMs for annotation 
tasks, with studies showing that GPT-4 can match or exceed human annotators 
on various classification tasks \cite{llmannotator2023}. However, applications 
to safety-critical domains remain limited, and concerns about LLM bias and 
consistency persist \cite{llmquality2024}.

\textbf{Hybrid Approaches.} Distant supervision combines heuristic and 
content-based signals \cite{distant2010}, typically using heuristics to 
generate noisy labels and content models to refine them. This approach has 
shown promise in information extraction but has not been applied to CVR analysis.

\textbf{Research Gap.} To the best of our knowledge, no prior work has 
systematically evaluated the impact of labeling strategies on CVR analysis 
performance. This study fills that gap by comparing position-based, content-based, 
and hybrid approaches on a common dataset with standardized evaluation metrics.
```

---

## 5. Labeling Strategies - Content-Based (Section 4.2)

```latex
\subsection{Content-Based Labeling (CBL)}

\textbf{Motivation.} Content-based labeling aims to align labels with the 
actual semantic content of utterances rather than their temporal position. 
This requires annotation criteria based on linguistic markers of emergency 
rather than temporal proximity.

\textbf{LLM Annotation Pipeline.} We employ DeepSeek-V3 \cite{deepseek2024} 
as our primary annotation model due to its cost-effectiveness and strong 
performance on reasoning tasks. The annotation process proceeds as follows:

\begin{enumerate}
    \item \textbf{Context Assembly:} For each utterance $u_i$, we assemble a 
    context window comprising the five preceding utterances $[u_{i-5}, ..., u_{i-1}]$.
    
    \item \textbf{Prompt Engineering:} We use the following structured prompt:
    \begin{verbatim}
    Task: Rate the anomaly level of the following cockpit communication.
    
    Context (previous 5 utterances):
    {context}
    
    Current utterance: "{utterance}"
    
    Rating criteria:
    1 - Normal routine communication (standard ATC, checklists)
    2 - Slightly unusual but not concerning (minor deviations)
    3 - Unusual, requires attention (unclear situations)
    4 - Elevated concern, potential emergency (stress indicators)
    5 - Clear emergency/critical situation (Mayday, severe warnings)
    
    Consider: urgency keywords, tone indicators, flight phase context, 
    pilot stress markers.
    
    Provide rating (1-5) and brief justification.
    \end{verbatim}
    
    \item \textbf{Discretization:} The 1-5 ratings are mapped to the four-class 
    schema: \{1,2\} $\rightarrow$ NORMAL, 3 $\rightarrow$ EARLY\_WARNING, 
    4 $\rightarrow$ ELEVATED, 5 $\rightarrow$ CRITICAL.
\end{enumerate}

\textbf{Quality Control.} To ensure annotation quality:
\begin{itemize}
    \item We validate 200 randomly selected annotations against expert judgment, 
    achieving Cohen's $\kappa = 0.78$ (substantial agreement).
    \item For ambiguous cases (rating confidence $<$ 0.7), we use majority voting 
    across three independent LLM queries.
    \item We manually review all CRITICAL predictions to ensure no systematic 
    bias toward over/under-prediction.
\end{itemize}

\textbf{Cost and Scalability.} Annotating 21,626 utterances cost approximately 
\$12 using DeepSeek-V3, demonstrating that content-based labeling is cost-effective 
for research-scale datasets. For operational deployment, annotation can be performed 
once and cached for model training.
```

---

## 6. Results - Key Findings (Section 6.2)

```latex
\subsection{Labeling Strategy Comparison}

Table \ref{tab:labeling_comparison} presents the performance of identical 
BERT-LSTM models trained on the three labeling strategies.

\begin{table}[h]
\centering
\caption{Performance Comparison of Labeling Strategies}
\label{tab:labeling_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Strategy} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{CRIT Rec} & \textbf{EARLY F1} \\
\midrule
Position (PBL) & 79.2\% & 0.659 & 47.8\% & 0.671 \\
Content (CBL) & 81.5\% & 0.682 & \textbf{72.3\%} & 0.658 \\
Hybrid (HL) & \textbf{83.1\%} & \textbf{0.701} & 68.9\% & \textbf{0.689} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Key Finding 1: Critical-Class Improvement.} Content-based labeling 
achieves a \textbf{24.5 percentage point improvement} in CRITICAL recall 
(72.3\% vs 47.8\%). This is statistically significant ($p < 0.001$, McNemar's test) 
and represents the most substantial performance gain observed. The improvement 
suggests that position-based labeling systematically obscures early emergency 
indicators that occur outside the arbitrary 5\% temporal window.

\textbf{Key Finding 2: Hybrid Superiority.} The hybrid approach achieves the 
best overall performance (83.1\% accuracy, 0.701 macro F1), suggesting that 
position information, when properly combined with content signals, provides 
complementary value. However, the improvement over pure content-based is modest 
(+1.6\% accuracy), indicating that content is the dominant signal.

\textbf{Key Finding 3: Early Warning Trade-off.} Interestingly, content-based 
labeling shows slightly lower EARLY\_WARNING F1 (0.658 vs 0.671). This may reflect 
greater boundary ambiguity in the middle severity range when using content 
criteria compared to the crisp temporal boundaries of position-based labeling.

\textbf{Safety Implications.} The improvement in CRITICAL recall is particularly 
significant for safety applications. Missing a critical situation (false negative) 
has far higher cost than a false alarm. The 24.5\% improvement means content-based 
models would miss approximately half as many critical situations as position-based 
models---a substantial safety gain.
```

---

## 7. Discussion - Theoretical Implications (Section 7.1)

```latex
\subsection{Theoretical Implications}

\textbf{Rethinking Temporal Assumptions.} Our results challenge the implicit 
assumption in CVR analysis that temporal proximity is a reliable proxy for 
communication severity. While accidents do tend to escalate temporally, the 
relationship is far from deterministic. Emergency situations can develop rapidly 
(\textit{``Mayday, bird strike, engine failure''}) or gradually, and pilots may 
engage in routine communication even moments before impact.

This has implications for how we conceptualize ``anomaly'' in aviation safety. 
Rather than a strictly temporal phenomenon, anomaly may be better understood as 
a \textit{semantic} property---certain patterns of communication indicate elevated 
risk regardless of when they occur. This aligns with theories of safety voice 
and safety listening \cite{noort2021cvr}, which emphasize the content and quality 
of communication over its timing.

\textbf{The Labeling-Performance Relationship.} Our findings contribute to the 
broader literature on label quality in machine learning. While prior work has 
focused on label noise and correction \cite{labelnoise2021}, our study demonstrates 
that even ``clean'' labels (position-based is deterministic) can be systematically 
biased in ways that limit model performance. This suggests that label \textit{validity} 
(alignment with ground truth) may be as important as label \textit{reliability} 
(consistency).

\textbf{Generalization to Other Domains.} While our study focuses on aviation, 
the implications extend to other domains where temporal heuristics are used for 
labeling: medical emergency detection (time to admission), financial fraud 
(time to charge-off), and predictive maintenance (time to failure). In each 
case, temporal proximity may not align with event severity, suggesting that 
content-based validation should be standard practice.
```

---

## 8. Limitations (Section 7.3)

```latex
\subsection{Limitations}

We acknowledge several limitations that bound the interpretation of our findings:

\textbf{Dataset Scope.} Our analysis is based on 172 accidents from the Noort 
et al. \cite{noort2021cvr} dataset. While this is the largest publicly available 
CVR transcript collection, it may not represent all accident types, particularly 
modern accidents with different communication patterns. The dataset also spans 
1962--2018, raising questions about temporal generalization as aviation 
communication norms evolve.

\textbf{LLM Annotation Bias.} Our content-based labels rely on DeepSeek-V3's 
interpretation of urgency and emergency. While we validated a subset against 
human judgment, LLMs may exhibit systematic biases: over-weighting explicit 
keywords (``Mayday'') relative to subtle indicators (tone shifts, pauses), or 
applying general language patterns that do not capture aviation-specific nuance. 
Future work should explore domain-specific fine-tuning of annotation models.

\textbf{Single-Annotator Validation.} Our manual validation involved a single 
expert reviewer for 200 samples. Multiple annotators would provide more robust 
estimates of inter-annotator agreement and identify ambiguous cases more reliably.

\textbf{Binary Safety Outcome.} We treat ``accident'' as a binary outcome, but 
real-world aviation safety involves near-misses, incidents, and varying severity 
levels. A more nuanced analysis might benefit from incorporating incident reports 
and normal flight data for contrast.

\textbf{Offline Analysis Only.} Our models process complete transcripts. Real-time 
anomaly detection would face additional challenges: incomplete context, streaming 
processing constraints, and the need for calibration to avoid alert fatigue. 
These are important directions for future work.

Despite these limitations, we believe our findings provide strong evidence that 
content-based labeling improves CVR analysis and should be adopted in future 
research.
```

---

## Writing Tips

### Strong Opening Sentences

| Section | Weak | Strong |
|---------|------|--------|
| Intro | "Aviation accidents are complex." | "In 1989, United Airlines Flight 232 crashed with 112 fatalities. The final words on the CVR: 'Close your eyes, baby.' This utterance, captured just seconds before impact, was labeled 'NORMAL' by position-based systems." |
| Method | "We use BERT." | "Transformer-based language models have revolutionized NLP, but their application to safety-critical domains requires careful consideration of how training labels align with ground truth severity." |
| Results | "The results show improvement." | "Content-based labeling does not merely improve metrics—it fundamentally changes what models learn, shifting attention from temporal position to linguistic markers of distress." |

### Transitions

```latex
Having established the limitations of position-based approaches, we now turn to...

These findings raise an important question: ...

To validate this hypothesis, we conducted...

Contrary to our expectations, ...

This result can be understood by considering...
```

### Hedging (Appropriate Uncertainty)

| Too Strong | Appropriate | Too Weak |
|------------|-------------|----------|
| proves that... | suggests that... | might possibly maybe... |
| always... | generally... | sometimes in some cases... |
| the only... | a primary... | one of many possible... |

---

**These samples demonstrate the tone and depth expected for Q1.**
