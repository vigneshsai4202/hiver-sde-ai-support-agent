from src.data.load import load_tweets
from src.data.clean import (
    clean_dataframe,
    cleaning_quality_report
)


INPUT_PATH = "data/raw/tweets.csv"
OUTPUT_PATH = "data/processed/cleaned_tweets.csv"


def main():
    print("Loading dataset...")

    df = load_tweets(INPUT_PATH)

    print(
        f"Rows loaded: {len(df):,}"
    )

    print("Cleaning dataset...")

    cleaned = clean_dataframe(df)

    print("\nCleaning quality report:\n")

    report = cleaning_quality_report(cleaned)

    for key, value in report.items():
        print(f"{key}: {value:,}")

    print("\nSaving cleaned dataset...")

    cleaned.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved cleaned dataset to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()