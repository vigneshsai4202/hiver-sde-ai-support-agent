import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import sys


from src.interactions.extract import (
    extract_support_interactions,
    interaction_summary,
)


INPUT_PATH = "data/processed/reconstructed_tweets.csv"
OUTPUT_PATH = "data/processed/support_interactions.csv"
SUMMARY_PATH = "reports/interaction_summary.csv"


def main():
    print("Loading reconstructed tweets...")

    usecols = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id",
        "conversation_id",
    ]

    df = pd.read_csv(
        INPUT_PATH,
        usecols=usecols,
        encoding="utf-8",
        dtype={
            "tweet_id": "string",
            "author_id": "string",
            "in_response_to_tweet_id": "string",
            "conversation_id": "string",
        },
        low_memory=False,
    )

    print(f"Rows loaded: {len(df):,}")

    print("Extracting direct customer -> support interactions...")

    interactions = extract_support_interactions(df)

    print(f"Interactions extracted: {len(interactions):,}")
    print(
        "Unique customer tweets:",
        f"{interactions['customer_tweet_id'].nunique():,}",
    )
    print(
        "Unique support tweets:",
        f"{interactions['support_tweet_id'].nunique():,}",
    )
    print(
        "Unique brands:",
        f"{interactions['brand'].nunique():,}",
    )

    print("\nResponse-time statistics:")

    print(
        interactions["response_time_minutes"]
        .describe()
        .to_string()
    )

    print("\nTop brands by direct interactions:")

    brand_counts = (
        interactions.groupby("brand")
        .size()
        .sort_values(ascending=False)
        .head(20)
    )

    print(brand_counts.to_string())

    print("\nSaving interactions...")

    interactions.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved to {OUTPUT_PATH}")

    summary = interaction_summary(interactions)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()