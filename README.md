# Hiver SDE Intern — AI Support Agent

AI support-agent prototype built for AppleSupport using the
Customer Support on Twitter dataset.

## What it does

Given a customer message, the system:

1. Classifies the primary support intent
2. Retrieves historically similar AppleSupport interactions
3. Generates a grounded draft response
4. Decides AUTO vs ESCALATE

## Quick Start

### 1. Install dependencies

pip install -r requirements.txt

### 2. Configure API key

Create `.env`:

GROQ_API_KEY=your_key_here

### 3. Run the example

python scripts/12_test_response.py

### 4. Reproduce evaluation

Run the AI classifier evaluation:

```bash
python scripts/10_evaluate_ai_classifier.py
