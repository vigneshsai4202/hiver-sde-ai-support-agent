# Response generation consumes the customer context, predicted intent,
# retrieved historical support evidence, and explicit safety constraints.
import os

from groq import Groq


SYSTEM_PROMPT = """
You are a customer support response assistant.

Draft a concise, helpful response to the customer's current issue.

You are given historical support interactions from the same brand.
Use them as evidence for how the brand historically handled similar
customer issues.
- Use plain text only.
- Do not use emojis, numbered emoji characters, markdown formatting, or special Unicode symbols.
- Use simple numbered lists such as "1.", "2.", "3." when needed.


IMPORTANT:
- Do not copy historical responses verbatim.
- Do not claim that a historical solution definitely resolved the issue.
- Do not invent policies, troubleshooting steps, refunds, guarantees,
  or technical facts that are not supported by the historical evidence.
- Prefer troubleshooting actions that appear in the historical examples.
- If historical evidence is insufficient, say that the issue needs
  further investigation or human support.
- Keep the response concise and suitable for a customer-support channel.
- Do not mention that you are an AI.
- Do not invent iOS versions, settings paths, troubleshooting steps, policies, or technical facts.
- Only recommend actions that are supported by the retrieved historical examples.
- If historical evidence is insufficient, keep the response cautious and ask for information or recommend human support.
"""


def create_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    return Groq(api_key=api_key)


def generate_response(
    customer_text,
    intent,
    historical_examples,
    model="openai/gpt-oss-20b",
):
    """
    Generate a response grounded in retrieved historical interactions.
    """

    evidence = []

    for example in historical_examples:

        evidence.append(
            f"""
Historical example {example['rank']}
Similarity: {example['similarity']:.4f}

Customer:
{example['customer_text']}

Historical AppleSupport response:
{example['support_text']}
"""
        )

    evidence_text = "\n".join(evidence)

    user_prompt = f"""
Current customer message:
{customer_text}

Predicted intent:
{intent}

Historical AppleSupport evidence:
{evidence_text}

Draft the best concise customer-support response for the current customer.
"""

    client = create_client()

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        reasoning_effort="low",
        include_reasoning=False,

        temperature=0,

        max_completion_tokens=256,

        stream=False,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "Groq returned empty response."
        )

    return content.strip()