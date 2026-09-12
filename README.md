# Hiver SDE Intern — AI Support Agent

AI support-agent prototype built for **AppleSupport** using the **Customer Support on Twitter** dataset.

The system learns from historically observed AppleSupport customer-support interactions and uses them as evidence for intent classification, response drafting, and escalation decisions.

## What it does

Given a customer message, the system:

1. Classifies the primary support intent
2. Retrieves historically similar AppleSupport interactions
3. Generates a grounded draft response
4. Decides whether to `AUTO` handle or `ESCALATE`
5. Provides a reason for the escalation decision

### High-level flow

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Retrieval
       |
       v
Grounded Response Generation
       |
       v
Escalation Decision
       |
       +----> AUTO
       |
       +----> ESCALATE
Selected Brand

The selected brand is AppleSupport.

The dataset contains customer-support interactions from many brands. I first audited the dataset, validated parent/response relationships, reconstructed conversation structure, and then extracted direct customer → support interactions.

For the final support-agent pipeline, AppleSupport was selected because it provides a sufficiently large set of historical support interactions.

The final AppleSupport interaction corpus contains approximately 32.8K direct customer → AppleSupport support interactions.

Intent Taxonomy

The system uses a 10-intent taxonomy consisting of nine primary support intents plus OTHER.

IOS_UPDATE
DEVICE_PERFORMANCE
BATTERY_POWER
CONNECTIVITY
APPS_APP_STORE
APPLE_ID_ICLOUD
MEDIA_SERVICES
PAYMENTS_PURCHASES
DATA_BACKUP_RESTORE
OTHER

The taxonomy is intentionally small so that the classifier focuses on the customer's primary support problem rather than attempting to model every possible issue.

Previous conversation context is also considered when classifying follow-up messages.

Historical Evidence

Historical customer-support interactions are used as evidence for how AppleSupport historically handled similar requests.

The system does not assume that a historical support reply proves that the customer's issue was resolved.

This distinction is important because a support reply may represent:

an initial troubleshooting step
a request for additional information
a request to continue the conversation privately
an escalation
or another support action

Historical responses are therefore treated as handling evidence rather than guaranteed solutions.

Quick Start
1. Clone the repository
git clone https://github.com/vigneshsai4202/hiver-sde-ai-support-agent.git
cd hiver-sde-ai-support-agent
2. Install dependencies
pip install -r requirements.txt

A virtual environment is recommended:

python -m venv venv

Windows:

venv\Scripts\activate

Then:

pip install -r requirements.txt
3. Configure the API key

Create a .env file in the project root:

GROQ_API_KEY=your_key_here

The API key is not committed to the repository.

The current implementation uses the Groq API with:

Model: openai/gpt-oss-20b
4. Run the end-to-end example
python scripts/12_test_response.py

This runs the complete pipeline:

Customer message
    ↓
Intent classification
    ↓
Historical retrieval
    ↓
Response generation
    ↓
Escalation decision

The example prints the predicted intent, confidence, retrieval evidence, generated response, escalation decision, and reason.

Evaluation

The repository contains a frozen 200-example golden set:

data/golden/golden_set_v1.csv

The examples were sampled from AppleSupport interactions across representative, hard, and edge cases and then AI-assisted labeled and manually reviewed.

Intent classification

Run:

python scripts/10_evaluate_ai_classifier.py

This evaluates the Groq-based classifier against the reviewed golden set.

Response generation

Run:

python scripts/14_evaluate_responses.py

This evaluates the response-generation pipeline using the golden examples, historical retrieval, and Groq response generation.

Escalation

Run:

python scripts/15_evaluate_escalation.py

This evaluates the AUTO vs ESCALATE decision against reviewed escalation labels.

LLM-as-judge

Run:

python scripts/17_llm_judge.py

The judge evaluates generated responses on:

relevance
groundedness
helpfulness
overall quality

A reviewer calibration subset is also used to measure agreement between the LLM judge and reviewer scores.

The full evaluation uses the Groq API and may be affected by provider rate/token limits. The reported results therefore distinguish successful predictions from unavailable provider calls.

Results

Results below are from the frozen 200-example evaluation set unless otherwise stated.

System	Accuracy	Macro F1
Majority baseline	17.0%	0.032
TF-IDF + Logistic Regression	75.0%*	0.651*
GPT-OSS-20B	61.5%*	0.543*

* The GPT-OSS-20B result is based on 195 successful predictions out of 200.

The TF-IDF result uses an 80/20 stratified split with a 40-example test set, so it is directional rather than a robust production estimate.

Escalation

The escalation evaluation achieved:

Accuracy:  70.2%
Precision: 66.7%
Recall:     6.9%
F1:         12.5%

The low escalation recall is an important limitation: the current policy is too conservative and misses many cases that reviewers marked for escalation.

LLM judge calibration

On a 25-example reviewer calibration subset:

Overall within ±1 point: 84%
Helpfulness within ±1:  84%
Relevance within ±1:    80%
Groundedness within ±1: 76%

Groundedness showed the largest disagreement between the reviewer and LLM judge, indicating that the judge can be overly optimistic about whether a response is actually supported by historical evidence.

The reviewer labels used for this calibration were AI-assisted and manually reviewed, so these numbers should be treated as calibration evidence rather than independent inter-annotator agreement.

Project Structure
hiver-sde-ai-support-agent/
│
├── configs/
│   ├── intent_taxonomy.yaml
│   └── llm.yaml
│
├── data/
│   ├── golden/
│   │   └── golden_set_v1.csv
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── README.md
│
├── evaluation/
│   └── README.md
│
├── reports/
│   ├── README.md
│   ├── applesupport_summary.csv
│   ├── brand_selection.csv
│   ├── conversation_summary.csv
│   └── interaction_summary.csv
│
├── scripts/
│   ├── 01_audit.py
│   ├── 02_clean.py
│   ├── 02_select_brand.py
│   ├── 03_reconstruct.py
│   ├── 04_extract_interactions.py
│   ├── 05_select_brand.py
│   ├── 06_inspect_intents.py
│   ├── 07_baseline_majority.py
│   ├── 08_baseline_tfidf.py
│   ├── 09_test_ai_classifier.py
│   ├── 10_evaluate_ai_classifier.py
│   ├── 11_test_retrieval.py
│   ├── 12_test_response.py
│   ├── 13_test_escalation.py
│   ├── 14_evaluate_responses.py
│   ├── 15_evaluate_escalation.py
│   ├── 16_prepare_judge_set.py
│   └── 17_llm_judge.py
│
├── src/
│   ├── baselines/
│   ├── brand_selection/
│   ├── classifier/
│   ├── conversations/
│   ├── data/
│   ├── escalation/
│   ├── evaluation/
│   ├── golden_set/
│   ├── intent_discovery/
│   ├── interactions/
│   ├── response/
│   └── retrieval/
│
├── tests/
│   └── test_clean.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── Hiver SDE Intern Report.pdf
Key Design Decisions
1. Direct customer → support interactions

Rather than treating every connected conversation component as one clean dialogue, the pipeline extracts direct customer → AppleSupport reply pairs.

This avoids incorrectly treating large announcement/reply threads as single customer conversations.

2. Context-aware intent classification

Previous context is provided for follow-up messages where available.

This helps distinguish cases where the latest message alone is ambiguous but clearly refers to an ongoing support issue.

3. TF-IDF retrieval

Historical interactions are retrieved using TF-IDF cosine similarity.

The retriever:

normalizes URLs and mentions
uses word bigrams
uses sublinear TF scaling
retrieves the most similar historical customer messages
4. Evidence-aware response generation

The response generator is instructed to use retrieved interactions as historical evidence and avoid:

copying historical replies verbatim
claiming that a historical action definitely solved the issue
inventing policies
inventing troubleshooting steps
inventing guarantees or technical facts
5. Conservative escalation policy

The current policy escalates when:

intent confidence is below the threshold
historical retrieval evidence is weak
the intent is payment/transaction related

This policy is intentionally simple and is one of the areas identified for improvement.

Known Limitations

The system is a prototype rather than a production support agent.

Important limitations include:

Escalation recall is low. The current rule-based policy misses many cases requiring human review.
Intent boundaries remain difficult. In particular, update-related problems can overlap with device performance and battery issues.
Retrieval similarity does not guarantee grounding. A highly similar historical example can still lead to an unsupported generated response.
The LLM judge can overestimate groundedness. Reviewer calibration showed weaker agreement on groundedness than on other dimensions.
Evaluation labels are AI-assisted and manually reviewed. They are not independent multi-annotator human labels.
Provider limits affect reproducibility. Groq rate/token limits can prevent every example from completing in one run.
The system does not perform real account, payment, refund, or backend actions.
Detailed Report

The complete take-home report is included in the repository:

Hiver SDE Intern Report.pdf

The report contains:

problem framing
dataset and interaction construction
intent taxonomy
baseline comparison
evaluation results
failure analysis
misleading headline-number analysis
one-more-week improvement plan
decision log
Scope

This project focuses on demonstrating an end-to-end AI support-agent architecture.

It does not attempt to build:

a production customer-support UI
a complete Apple knowledge base
autonomous account actions
payment/refund execution
guaranteed issue resolution
a full-scale production deployment

The goal is to demonstrate how historical support interactions can be combined with intent classification, retrieval, response generation, and escalation into a measurable support-agent workflow.
