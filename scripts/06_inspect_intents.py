import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


INPUT_PATH = "data/processed/applesupport_interactions.csv"


def main():

    print("Loading AppleSupport interactions...")

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8",
        low_memory=False,
    )

    print(f"Rows: {len(df):,}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample customer messages:\n")

    sample = df.sample(
        n=min(50, len(df)),
        random_state=42,
    )

    for i, row in enumerate(
        sample[
            [
                "interaction_id",
                "customer_text",
                "support_text",
            ]
        ].itertuples(index=False),
        start=1,
    ):

        print("=" * 80)
        print(f"Example {i}")
        print(f"Interaction: {row.interaction_id}")
        print(f"\nCUSTOMER:\n{row.customer_text}")
        print(f"\nAPPLE SUPPORT:\n{row.support_text}")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()