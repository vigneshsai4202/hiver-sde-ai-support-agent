import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.classifier.classify import classify_message
from src.retrieval.retrieve import HistoricalRetriever
from src.escalation.decision import decide_escalation


def main():

    print("=" * 60)
    print("ESCALATION DECISION TEST")
    print("=" * 60)

    customer_message = (
        "My iPhone battery is draining very quickly "
        "after the latest iOS update."
    )

    # Intent
    classification = classify_message(
        customer_text=customer_message
    )

    # Retrieval
    retriever = HistoricalRetriever(
        "data/processed/applesupport_interactions.csv"
    )

    historical_examples = retriever.retrieve(
        customer_message,
        top_k=5,
    )

    # Escalation
    decision = decide_escalation(
        intent=classification["intent"],
        confidence=classification["confidence"],
        historical_examples=historical_examples,
    )

    print("\nCustomer:")
    print(customer_message)

    print("\nClassification:")
    print(classification)

    print("\nEscalation:")
    print(decision)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()