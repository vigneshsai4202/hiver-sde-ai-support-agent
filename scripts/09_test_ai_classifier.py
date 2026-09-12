import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.classifier.classify import classify_message


GOLDEN_PATH = "data/golden/golden_set_v1.csv"


def main():

    df = pd.read_csv(
        GOLDEN_PATH,
        encoding="utf-8",
    )

    # Use one real reviewed example.
    row = df.iloc[0]

    print("=" * 60)
    print("AI CLASSIFIER TEST")
    print("=" * 60)

    print("\nCustomer:")
    print(row["text"])

    print("\nPrevious context:")
    print(row["previous_context"])

    print("\nExpected human intent:")
    print(row["human_intent"])

    print("\nCalling Groq...")

    result = classify_message(
        customer_text=row["text"],
        previous_context=row["previous_context"],
    )

    print("\nAI prediction:")
    print(result)


if __name__ == "__main__":
    main()