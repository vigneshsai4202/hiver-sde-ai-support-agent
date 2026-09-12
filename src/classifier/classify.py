import os
import re

from groq import Groq


INTENTS = [
    "IOS_UPDATE",
    "DEVICE_PERFORMANCE",
    "BATTERY_POWER",
    "CONNECTIVITY",
    "APPS_APP_STORE",
    "APPLE_ID_ICLOUD",
    "MEDIA_SERVICES",
    "PAYMENTS_PURCHASES",
    "DATA_BACKUP_RESTORE",
    "OTHER",
]


SYSTEM_PROMPT = """
You are a customer-support intent classifier.

Classify the PRIMARY support problem.

Use previous context when available.

INTENTS:

IOS_UPDATE:
Problems installing, obtaining, updating, or troubleshooting an iOS update.

DEVICE_PERFORMANCE:
Crashes, freezing, restarting, boot problems, severe slowness, or general
device malfunction.

BATTERY_POWER:
Battery drain, battery life, charging, power, or device dying because of
battery/power problems.

CONNECTIVITY:
Wi-Fi, Bluetooth, cellular, network, signal, or connection problems.

APPS_APP_STORE:
App Store, application installation/download, or generic application problems.

APPLE_ID_ICLOUD:
Apple ID, authentication, sign-in, account access, or general iCloud
account problems.

MEDIA_SERVICES:
Apple Music, Podcasts, iTunes media, playback, or media-library problems.

PAYMENTS_PURCHASES:
Charges, billing, payments, purchases, refunds, or transaction problems.

DATA_BACKUP_RESTORE:
Backup, restore, recovery, synchronization, or data/photo recovery problems.

OTHER:
The problem genuinely does not fit the taxonomy or there is insufficient
information.
PRIMARY INTENT PRIORITY:

When the current message is a follow-up and previous context is provided,
identify the ongoing support problem before classifying the new symptom.

If the customer says "also", "it", "this", "still", "again", or otherwise
clearly refers to the previous problem, treat the previous context as part
of the same issue.

For example:

Previous context:
"Phone keeps going black, showing a loading wheel, then asking for a
passcode."

Current message:
"It's also draining my battery."

Classify as DEVICE_PERFORMANCE because the battery drain is an additional
symptom of the ongoing device malfunction, not the primary support issue.

Do NOT automatically choose an intent merely because the latest message
contains a strong keyword.

IMPORTANT RULES:

1. Choose exactly ONE primary intent.
2. Use previous context to understand vague follow-up messages.
3. Do not classify based only on keywords.
4. If an update is mentioned but the actual problem is battery drain,
   choose BATTERY_POWER.
5. If an update is mentioned but the actual problem is crashing, freezing,
   restarting, or severe slowness, choose DEVICE_PERFORMANCE.
6.  If multiple symptoms exist, choose the primary problem that best explains
   the ongoing support interaction. A newly mentioned secondary symptom does
   not automatically replace the primary intent.
7. If there is genuinely insufficient information, choose OTHER.

Return exactly these two lines:

INTENT: <intent>
CONFIDENCE: <number between 0 and 1>

Do not explain your answer.
"""


def create_client():
    """Create the Groq API client."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add GROQ_API_KEY=your_key to the .env file."
        )

    return Groq(api_key=api_key)


def _parse_response(content):
    """
    Parse the model's short text response.

    Expected format:

    INTENT: DEVICE_PERFORMANCE
    CONFIDENCE: 0.95
    """

    if not content:
        raise ValueError(
            "Groq returned empty content."
        )

    content = content.strip()

    intent_match = re.search(
        r"INTENT\s*:\s*([A-Z_]+)",
        content,
        flags=re.IGNORECASE,
    )

    confidence_match = re.search(
        r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)",
        content,
        flags=re.IGNORECASE,
    )

    if not intent_match:
        raise ValueError(
            f"Could not parse intent from Groq response: {content}"
        )

    if not confidence_match:
        raise ValueError(
            f"Could not parse confidence from Groq response: {content}"
        )

    intent = intent_match.group(1).upper()

    if intent not in INTENTS:
        raise ValueError(
            f"Invalid intent returned by Groq: {intent}"
        )

    confidence = float(
        confidence_match.group(1)
    )

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return {
        "intent": intent,
        "confidence": confidence,
    }


def classify_intent(
    client,
    customer_text,
    previous_context="",
    model="openai/gpt-oss-20b",
):
    """
    Classify one customer-support message.

    GPT-OSS is a reasoning model. We explicitly disable reasoning output
    because this task only needs the final classification.
    """

    customer_text = (
        ""
        if customer_text is None
        else str(customer_text)
    )

    previous_context = (
        ""
        if previous_context is None
        else str(previous_context)
    )

    user_prompt = f"""
Customer message:
{customer_text}

Previous context:
{previous_context if previous_context.strip() else "(none)"}

Return exactly:

INTENT: <intent>
CONFIDENCE: <number>
"""

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

        # GPT-OSS reasoning configuration.
        # We only need the final answer for classification.
        reasoning_effort="low",
        include_reasoning=False,

        temperature=0,

        # Use the current Groq parameter name.
        max_completion_tokens=128,

        stream=False,
    )

    message = response.choices[0].message

    content = message.content

    if not content:
        raise ValueError(
            "Groq returned empty content. "
            f"finish_reason={response.choices[0].finish_reason}"
        )

    return _parse_response(content)


def classify_message(
    customer_text,
    previous_context="",
    model="openai/gpt-oss-20b",
):
    """
    Backward-compatible helper used by scripts/09_test_ai_classifier.py.
    """

    client = create_client()

    return classify_intent(
        client=client,
        customer_text=customer_text,
        previous_context=previous_context,
        model=model,
    )