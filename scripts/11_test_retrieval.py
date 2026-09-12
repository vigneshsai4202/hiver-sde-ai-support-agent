from src.retrieval.retrieve import HistoricalRetriever


def main():

    print("=" * 60)
    print("HISTORICAL RETRIEVAL TEST")
    print("=" * 60)

    retriever = HistoricalRetriever(
        "data/processed/applesupport_interactions.csv"
    )

    customer_message = (
        "My iPhone battery is draining very quickly "
        "after the latest iOS update."
    )

    print("\nCustomer:")
    print(customer_message)

    print("\nTop historical matches:\n")

    results = retriever.retrieve(
        customer_message,
        top_k=5,
    )

    for result in results:

        print("-" * 60)

        print(
            f"Rank: {result['rank']} | "
            f"Similarity: {result['similarity']:.4f}"
        )

        print("\nHistorical customer:")
        print(result["customer_text"])

        print("\nHistorical AppleSupport response:")
        print(result["support_text"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()