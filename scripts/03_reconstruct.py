from src.data.load import load_tweets
from src.conversations.reconstruct import (
    reconstruct_conversations,
    conversation_summary,
)


INPUT_PATH = "data/processed/tweets_clean.csv"

CONVERSATIONS_PATH = (
    "data/processed/reconstructed_tweets.csv"
)

SUMMARY_PATH = (
    "reports/conversation_summary.csv"
)


def main():
    print("Loading cleaned dataset...")

    df = load_tweets(INPUT_PATH)

    print(
        f"Rows loaded: {len(df):,}"
    )

    print("Reconstructing conversations...")

    reconstructed = reconstruct_conversations(df)

    print(
        f"Tweets reconstructed: "
        f"{len(reconstructed):,}"
    )

    print(
        "Number of conversations:",
        f"{reconstructed['conversation_id'].nunique():,}"
    )

    print("Building conversation summary...")

    summary = conversation_summary(
        reconstructed
    )

    print(
        f"Conversation records: "
        f"{len(summary):,}"
    )

    print("\nConversation summary statistics:")

    print(
        summary[
            [
                "tweet_count",
                "customer_turns",
                "support_turns",
                "duration_minutes",
            ]
        ].describe().to_string()
    )

    print("\nTwo-sided conversations:")

    print(
        int(summary["is_two_sided"].sum())
    )

    print("\nSaving reconstructed tweets...")

    reconstructed.to_csv(
        CONVERSATIONS_PATH,
        index=False
    )

    print(
        f"Saved to {CONVERSATIONS_PATH}"
    )

    print("\nSaving conversation summary...")

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print(
        f"Saved to {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()