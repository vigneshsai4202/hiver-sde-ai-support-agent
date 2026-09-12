import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.classifier.classify import classify_message
from src.retrieval.retrieve import HistoricalRetriever
from src.response.generate import generate_response
from src.escalation.decision import decide_escalation


def main():

    print("=" * 60)
    print("END-TO-END AI SUPPORT AGENT TEST")
    print("=" * 60)

    customer_message = (
        "My iPhone battery is draining very quickly "
        "after the latest iOS update."
    )

    print("\nCustomer:")
    print(customer_message)

    # ---------------------------------------------------------
    # 1. Intent Classification
    # ---------------------------------------------------------
    print("\nClassifying intent...")

    classification = classify_message(
        customer_text=customer_message
    )

    intent = classification["intent"]
    confidence = classification["confidence"]

    print("\nIntent:")
    print(intent)

    print("\nConfidence:")
    print(f"{confidence:.2f}")

    # ---------------------------------------------------------
    # 2. Historical Retrieval
    # ---------------------------------------------------------
    print("\nRetrieving historical AppleSupport interactions...")

    retriever = HistoricalRetriever(
        "data/processed/applesupport_interactions.csv"
    )

    historical_examples = retriever.retrieve(
        customer_message,
        top_k=5,
    )

    print(
        f"Retrieved {len(historical_examples)} historical examples."
    )

    if historical_examples:
        best_similarity = max(
            example["similarity"]
            for example in historical_examples
        )

        print(
            f"Best historical similarity: "
            f"{best_similarity:.2f}"
        )

    # ---------------------------------------------------------
    # 3. Response Generation
    # ---------------------------------------------------------
    print("\nGenerating grounded response...")

    response = generate_response(
        customer_text=customer_message,
        intent=intent,
        historical_examples=historical_examples,
    )

    print("\n" + "=" * 60)
    print("GENERATED RESPONSE")
    print("=" * 60)

    print(response)

    # ---------------------------------------------------------
    # 4. Escalation Decision
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("ESCALATION DECISION")
    print("=" * 60)

    escalation = decide_escalation(
        intent=intent,
        confidence=confidence,
        historical_examples=historical_examples,
    )

    print("\nDecision:")
    print(escalation["decision"])

    print("\nReason:")
    print(escalation["reason"])

    print("\n" + "=" * 60)
    print("END-TO-END TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()