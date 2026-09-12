from src.data.load import load_tweets
from src.data.clean import clean_dataframe
from src.brand_selection.select_brand import (
    calculate_brand_metrics,
    save_brand_report
)


INPUT_PATH = "data/raw/tweets.csv"
OUTPUT_PATH = "reports/brand_selection.csv"


def main():
    print("Loading dataset...")

    df = load_tweets(INPUT_PATH)

    print("Rows loaded:", f"{len(df):,}")

    print("Cleaning text and timestamps...")

    df = clean_dataframe(df)

    print("Calculating brand metrics...")

    metrics = calculate_brand_metrics(df)

    save_brand_report(
        metrics,
        OUTPUT_PATH
    )

    print("\nTop 30 brands:\n")

    print(
        metrics.head(30).to_string(index=False)
    )

    print(
        f"\nSaved report to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()