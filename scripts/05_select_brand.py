import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.brand_selection.filter_brand import (
    filter_brand_interactions,
    brand_summary,
)


INPUT_PATH = "data/processed/support_interactions.csv"
OUTPUT_PATH = "data/processed/applesupport_interactions.csv"
SUMMARY_PATH = "reports/applesupport_summary.csv"

BRAND = "AppleSupport"


def main():

    print("Loading support interactions...")

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8",
        low_memory=False,
    )

    print(f"Interactions loaded: {len(df):,}")

    print(f"\nSelecting brand: {BRAND}")

    apple = filter_brand_interactions(
        df,
        brand=BRAND,
    )

    print(f"AppleSupport interactions: {len(apple):,}")

    summary = brand_summary(apple)

    print("\nAppleSupport summary:")

    for key, value in summary.items():

        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value:,}")

    print("\nSaving AppleSupport interactions...")

    apple.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved to {OUTPUT_PATH}")

    pd.DataFrame(
        [summary]
    ).to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()