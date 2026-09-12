AUTO_THRESHOLD = 0.85
RETRIEVAL_THRESHOLD = 0.30


HIGH_RISK_INTENTS = {
    "PAYMENTS_PURCHASES",
}


def decide_escalation(
    intent,
    confidence,
    historical_examples,
):
    """
    Decide whether the generated response can be auto-handled
    or should be escalated to a human agent.
    """

    reasons = []

    # 1. Low classifier confidence
    if confidence < AUTO_THRESHOLD:
        reasons.append(
            f"low intent confidence ({confidence:.2f})"
        )

    # 2. No useful historical evidence
    if not historical_examples:
        reasons.append(
            "no historical support evidence retrieved"
        )

    else:
        best_similarity = max(
            example["similarity"]
            for example in historical_examples
        )

        if best_similarity < RETRIEVAL_THRESHOLD:
            reasons.append(
                f"weak historical similarity ({best_similarity:.2f})"
            )

    # 3. High-risk category
    if intent in HIGH_RISK_INTENTS:
        reasons.append(
            "payment or transaction issue requires human review"
        )

    if reasons:
        return {
            "decision": "ESCALATE",
            "reason": "; ".join(reasons),
        }

    return {
        "decision": "AUTO",
        "reason": (
            f"high intent confidence ({confidence:.2f}) "
            "and sufficient historical support evidence"
        ),
    }