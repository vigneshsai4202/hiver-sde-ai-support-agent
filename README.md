# AI Support Agent for AppleSupport

An end-to-end prototype AI support agent that classifies customer support requests, retrieves historical evidence, generates grounded responses, and makes escalation decisions. Built with historical AppleSupport customer-support interactions from Twitter.

## Overview

The system processes incoming customer messages through a five-stage pipeline:

1. **Intent Classification** — Classify the customer's support problem into one of 10 intents (e.g., battery drain, connectivity, device performance)
2. **Historical Retrieval** — Retrieve similar historical AppleSupport interactions as evidence for how similar problems were handled
3. **Grounded Response Generation** — Generate a response grounded in retrieved historical examples
4. **Escalation Decision** — Decide whether to handle automatically or escalate to human review
5. **Reasoning** — Provide a transparent reason for the escalation decision

```
Customer Message
       ↓
Intent Classification → (confidence score)
       ↓
Historical Retrieval → (top-k similar interactions)
       ↓
Response Generation → (grounded in historical evidence)
       ↓
Escalation Decision → AUTO or ESCALATE
       ↓
Escalation Reason
```

## Problem Framing

**Intent Classification**  
Map each customer message to one of 10 support intents. Use previous conversation context when available to disambiguate follow-up messages.

**Historical Evidence**  
Support replies are treated as handling *evidence*, not proof of resolution. A historical response may represent:
- An initial troubleshooting step
- A request for additional information
- A request to escalate privately
- Another form of support action

**Escalation Decision**  
Determine whether the system can AUTO handle the request or should ESCALATE to a human agent. This is a high-recall task—missing an escalation is costly.

## Selected Brand: AppleSupport

The system was trained on interactions from **AppleSupport**, selected from the Customer Support on Twitter dataset because it provides a sufficiently large and representative corpus of technical support interactions.

The final AppleSupport interaction corpus contains approximately **32.8K direct customer → AppleSupport support interactions**.

## Intent Taxonomy

A 10-intent taxonomy ensures the classifier focuses on the customer's primary support problem:

| Intent | Description |
|--------|-------------|
| `IOS_UPDATE` | Installing, obtaining, or troubleshooting an operating-system update |
| `DEVICE_PERFORMANCE` | Crashes, freezing, restarting, lag, severe slowness, or general device malfunction |
| `BATTERY_POWER` | Battery drain, charging, or power-related problems |
| `CONNECTIVITY` | Wi-Fi, Bluetooth, cellular, or other connection/network problems |
| `APPS_APP_STORE` | App Store, app installation/download, or generic application problems |
| `APPLE_ID_ICLOUD` | Apple ID, authentication, sign-in, or general iCloud account access |
| `MEDIA_SERVICES` | Apple Music, Podcasts, iTunes/media playback or library problems |
| `PAYMENTS_PURCHASES` | Charges, billing, payments, purchases, or refunds |
| `DATA_BACKUP_RESTORE` | Backup, restore, recovery, synchronization, or data-recovery problems |
| `OTHER` | Does not fit the taxonomy or has insufficient information |

## Historical Retrieval

Historical customer-support interactions are retrieved using **TF-IDF cosine similarity** on the customer's message.

The retriever:
- Normalizes URLs and mentions
- Uses word bigrams
- Applies sublinear TF scaling
- Retrieves the most similar historical customer messages and their corresponding AppleSupport replies

Retrieved examples inform response generation but do not guarantee grounding—a similar historical example can still lead to an unsupported generated response.

## Response Generation

Responses are generated using **Groq** with the **openai/gpt-oss-20b** model.

The generator is instructed to:
- Use retrieved interactions as historical evidence
- Avoid copying historical replies verbatim
- Avoid claiming a historical action definitely solved the issue
- Avoid inventing policies, troubleshooting steps, guarantees, or technical facts

## Escalation Decision

The current rule-based escalation policy escalates when:
- Intent confidence is below a threshold
- Historical retrieval evidence is weak
- The intent is payment/transaction related (`PAYMENTS_PURCHASES`)

**Note:** The escalation recall is deliberately conservative (6.9%), which means the policy intentionally errs toward human escalation. This is appropriate for high-stakes support but is not well-optimized for balancing precision and recall.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/vigneshsai4202/hiver-sde-ai-support-agent.git
cd hiver-sde-ai-support-agent
```

### 2. Set up Python environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

The API key is not committed to the repository.

### 5. Run the end-to-end example

```bash
python scripts/12_test_response.py
```

This script:
- Takes a sample customer message
- Classifies intent
- Retrieves historical examples
- Generates a response
- Makes an escalation decision
- Prints the full pipeline output

## Evaluation

The repository includes a frozen 200-example golden evaluation set in `data/golden/golden_set_v1.csv`. Examples were sampled across representative, hard, and edge cases, then AI-assisted labeled and manually reviewed.

**Note:** Golden labels are AI-assisted and manually reviewed, not independent multi-annotator ground truth.

### Intent Classification

```bash
python scripts/10_evaluate_ai_classifier.py
```

Evaluates the Groq-based classifier against the golden set using accuracy and macro F1.

### Baseline: Majority Class

```bash
python scripts/07_baseline_majority.py
```

Evaluates a majority-class baseline (always predicting the most common intent).

### Baseline: TF-IDF + Logistic Regression

```bash
python scripts/08_baseline_tfidf.py
```

Evaluates a traditional ML baseline using TF-IDF features and logistic regression on an 80/20 stratified split.

### Response Generation

```bash
python scripts/14_evaluate_responses.py
```

Evaluates the full response-generation pipeline on golden examples.

### Escalation Decision

```bash
python scripts/15_evaluate_escalation.py
```

Evaluates AUTO vs ESCALATE decisions against reviewed escalation labels.

### LLM-as-Judge

```bash
python scripts/17_llm_judge.py
```

Uses an LLM to evaluate generated responses on relevance, groundedness, helpfulness, and overall quality. Includes calibration on 25-example reviewer subset.

## Results

All results are from the frozen 200-example golden evaluation set unless otherwise stated.

### Intent Classification Accuracy

| System | Accuracy | Macro F1 | Notes |
|--------|----------|----------|-------|
| Majority baseline | 17.0% | 0.032 | Predicts most common intent |
| TF-IDF + Logistic Regression | 75.0% | 0.651 | 80/20 stratified split; directional, not robust |
| GPT-OSS-20B (Groq) | 61.5% | 0.543 | 195/200 successful; small reviewed set |

**Important caveat:** The 61.5% accuracy should not be interpreted as production performance. It is based on a 200-example reviewed evaluation set with AI-assisted labels. Results may vary significantly on different distributions or with different annotators.

### Escalation Decision

| Metric | Value |
|--------|-------|
| Accuracy | 70.2% |
| Precision | 66.7% |
| Recall | 6.9% |
| F1 | 12.5% |

**Critical limitation:** Low recall (6.9%) indicates the current policy is overly conservative and misses many cases that reviewers marked for escalation. The high accuracy is misleading—it reflects the class imbalance (most examples are not escalations) rather than strong escalation performance.

### LLM Judge Calibration

On a 25-example reviewer calibration subset:

| Dimension | Agreement (±1 point) |
|-----------|----------------------|
| Overall | 84% |
| Helpfulness | 84% |
| Relevance | 80% |
| Groundedness | 76% |

Groundedness showed the largest disagreement, indicating the LLM judge can be overly optimistic about whether responses are grounded in historical evidence.

## Failure Analysis

### 1. Escalation Recall is Very Low

The current rule-based policy misses many cases that should be escalated to humans. Improving escalation recall is the highest priority.

### 2. Intent Boundaries Remain Difficult

Update-related problems often overlap with device performance and battery issues. The taxonomy is small by design, but boundary cases remain hard to classify.

### 3. Retrieval Similarity Does Not Guarantee Grounding

A highly similar historical example can still lead to an unsupported or inaccurate generated response. Grounding validation is weak.

### 4. LLM Judge Overestimates Groundedness

The LLM judge agrees with human reviewers on groundedness only 76% of the time, versus 80%+ on other dimensions. The judge tends to be optimistic.

### 5. Provider Limits Affect Reproducibility

Groq rate and token limits can prevent every example from completing in a single evaluation run. Results distinguish successful from unavailable provider calls.

## Why 61.5% Should Not Be Treated as Production Accuracy

**Specific Limitations:**

- **Small evaluation set:** 200 examples is sufficient for directional evaluation, not production benchmarking
- **AI-assisted labels:** Golden labels are AI-generated and manually reviewed, not independent human ground truth
- **Provider limitations:** 195 out of 200 successful predictions; failed calls were skipped
- **Imbalanced classes:** Some intents (e.g., `OTHER`) may be under-represented
- **Directional comparison:** The TF-IDF baseline (75%) is also directional—an 80/20 split on 40 test examples
- **No production dataset:** Evaluation is only on the frozen golden set; real-world distribution is unknown
- **Low escalation recall:** The complementary escalation recall of 6.9% reveals the system is overly conservative

**Conclusion:** The 61.5% is a useful directional signal that the Groq-based classifier performs worse than the TF-IDF baseline on this small task. It should not be extrapolated to production performance.

## One More Week: Improvement Priorities

### 1. Improve Escalation Recall

Redesign the escalation policy to catch more cases that require human review. Current threshold-based rules are too simple.

### 2. Improve Retrieval

Experiment with dense retrieval (embeddings) and retrieval re-ranking to better surface relevant historical examples.

### 3. Improve Response Grounding

Add validation steps to check that generated responses are actually grounded in retrieved evidence before returning them.

### 4. Strengthen Evaluation

Collect independent multi-annotator human labels for a subset of golden examples to establish true inter-annotator agreement and reduce AI-assist bias.

### 5. Improve Reliability

Add error handling for provider rate limits, implement retry logic, and set up comprehensive logging for debugging.

## Project Structure

```
hiver-sde-ai-support-agent/
│
├── configs/
│   ├── intent_taxonomy.yaml          # 10-intent taxonomy definitions
│   └── llm.yaml                       # LLM configuration
│
├── data/
│   ├── golden/
│   │   └── golden_set_v1.csv          # 200-example frozen evaluation set
│   ├── raw/                           # Raw dataset (not included)
│   ├── processed/                     # Processed interactions
│   └── README.md
│
├── evaluation/
│   └── README.md
│
├── reports/
│   ├── README.md
│   ├── applesupport_summary.csv       # Brand summary stats
│   ├── brand_selection.csv            # Brand selection analysis
│   ├── conversation_summary.csv       # Conversation structure
│   └── interaction_summary.csv        # Interaction corpus stats
│
├── scripts/
│   ├── 01_audit.py                    # Audit raw dataset
│   ├── 02_clean.py                    # Clean and validate data
│   ├── 02_select_brand.py             # Brand selection analysis
│   ├── 03_reconstruct.py              # Reconstruct conversation threads
│   ├── 04_extract_interactions.py     # Extract customer → support pairs
│   ├── 05_select_brand.py             # Final brand selection
│   ├── 06_inspect_intents.py          # Inspect intent distribution
│   ├── 07_baseline_majority.py        # Majority baseline evaluation
│   ├── 08_baseline_tfidf.py           # TF-IDF baseline evaluation
│   ├── 09_test_ai_classifier.py       # Quick test of AI classifier
│   ├── 10_evaluate_ai_classifier.py   # Full AI classifier evaluation
│   ├── 11_test_retrieval.py           # Quick test of retrieval
│   ├── 12_test_response.py            # End-to-end pipeline test
│   ├── 13_test_escalation.py          # Quick test of escalation
│   ├── 14_evaluate_responses.py       # Response generation evaluation
│   ├── 15_evaluate_escalation.py      # Escalation evaluation
│   ├── 16_prepare_judge_set.py        # Prepare set for LLM judge
│   └── 17_llm_judge.py                # LLM-as-judge evaluation
│
├── src/
│   ├── baselines/                     # Baseline implementations
│   ├── brand_selection/               # Brand selection logic
│   ├── classifier/                    # Intent classification
│   ├── conversations/                 # Conversation reconstruction
│   ├── data/                          # Data loading and processing
│   ├── escalation/                    # Escalation decision logic
│   ├── evaluation/                    # Evaluation utilities
│   ├── golden_set/                    # Golden set utilities
│   ├── intent_discovery/              # Intent taxonomy discovery
│   ├── interactions/                  # Interaction extraction
│   ├── response/                      # Response generation
│   └── retrieval/                     # Historical retrieval
│
├── tests/
│   └── test_clean.py                  # Unit tests for data cleaning
│
├── .gitignore
├── requirements.txt
├── README.md
└── Hiver SDE Intern Report.pdf        # Detailed take-home report
```

## Key Design Decisions

### 1. Direct Customer → Support Interactions

Rather than treating every connected conversation component as a single clean dialogue, the pipeline extracts direct customer → AppleSupport reply pairs. This avoids incorrectly modeling large Twitter announcement threads as single customer conversations.

### 2. Context-Aware Intent Classification

Previous conversation context is provided for follow-up messages when available. This helps disambiguate cases where the latest message alone is ambiguous but clearly refers to an ongoing support issue.

### 3. TF-IDF Retrieval

Historical interactions are retrieved using TF-IDF cosine similarity because:
- It is interpretable and efficient
- It scales to the corpus size
- Retrieved examples are human-readable for validation

### 4. Evidence-Aware Response Generation

The response generator is instructed to use retrieved interactions as historical evidence and avoid:
- Copying historical replies verbatim
- Claiming a historical action definitely solved the issue
- Inventing policies, troubleshooting steps, or technical facts

### 5. Conservative Escalation Policy

The current policy is intentionally conservative—it escalates when confidence is low, evidence is weak, or the intent is payment-related. Missing an escalation is treated as worse than over-escalating.

## Data and Reproducibility

### Included

- `data/golden/golden_set_v1.csv` — Frozen 200-example evaluation set (AI-assisted labels, manually reviewed)
- `configs/intent_taxonomy.yaml` — 10-intent taxonomy
- `src/` — All source code for the pipeline
- `scripts/` — All evaluation and test scripts
- `Hiver SDE Intern Report.pdf` — Detailed analysis and findings

### Excluded

- The raw Twitter Customer Support dataset (not included due to size and licensing)
- The full AppleSupport interaction corpus is processed but `data/raw/` is not committed

### Running Evaluation

To run the full evaluation pipeline, you will need:
1. Groq API key (`.env` file)
2. Processed AppleSupport interactions (`data/processed/applesupport_interactions.csv`)

If the processed data is missing, you can reconstruct it by running the data pipeline scripts in order:
```bash
python scripts/01_audit.py
python scripts/02_clean.py
python scripts/03_reconstruct.py
python scripts/04_extract_interactions.py
```

## Limitations

This is a prototype, not a production support agent. Important limitations include:

- **Escalation recall is very low (6.9%)** — The current rule-based policy misses many cases requiring human review
- **Intent boundaries are difficult** — Update-related problems overlap with device performance and battery issues
- **Retrieval does not guarantee grounding** — Similar historical examples can still lead to unsupported responses
- **LLM judge overestimates groundedness** — Reviewer calibration showed weaker agreement on this dimension
- **Evaluation labels are AI-assisted** — Not independent multi-annotator ground truth
- **Provider rate limits** — Evaluation runs can be interrupted by Groq rate/token limits
- **No autonomous actions** — The system does not perform real account, payment, refund, or backend operations
- **No production UI** — This is a backend prototype, not a customer-facing system

## Detailed Report

The complete analysis is in:

**[Hiver SDE Intern Report.pdf](https://github.com/vigneshsai4202/hiver-sde-ai-support-agent/blob/main/Hiver%20SDE%20Intern%20Report.pdf)**

The report includes:
- Detailed problem framing and dataset analysis
- Brand selection justification and interaction corpus construction
- Baseline design and results
- Full evaluation results with confidence intervals
- Failure mode analysis and case studies
- Misleading headline-number analysis (why 61.5% is not production accuracy)
- One-more-week improvement plan
- Complete decision log

## Repository

[github.com/vigneshsai4202/hiver-sde-ai-support-agent](https://github.com/vigneshsai4202/hiver-sde-ai-support-agent)

---

**Submission for Hiver SDE Intern Role**
