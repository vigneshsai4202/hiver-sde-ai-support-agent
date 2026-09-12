from src.data.load import load_tweets
from src.data.clean import (
    clean_dataframe,
    cleaning_quality_report,
)


INPUT_PATH = "data/raw/tweets.csv"
OUTPUT_PATH = "data/processed/tweets_clean.csv"


def main():
    print("Loading dataset...")
    df = load_tweets(INPUT_PATH)

    print(f"Rows loaded: {len(df):,}")

    print("Cleaning dataset...")
    cleaned_df = clean_dataframe(df)

    print("\nCleaning quality report:")
    print("-" * 30)

    report = cleaning_quality_report(cleaned_df)

    for key, value in report.items():
        print(f"{key}: {value:,}")

    print("\nWriting cleaned dataset...")

    cleaned_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"\nWrote {len(cleaned_df):,} rows")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()