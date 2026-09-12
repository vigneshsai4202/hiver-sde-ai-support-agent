import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.baselines.majority import majority_class_predict


GOLDEN_PATH = "data/golden/golden_set_v1.csv"


def main():

    print("Loading golden dataset...")

    df = pd.read_csv(
        GOLDEN_PATH,
        encoding="utf-8",
    )

    print(f"Golden examples: {len(df):,}")

    # ---------------------------------------------------------
    # Validate labels
    # ---------------------------------------------------------

    if df["human_intent"].isna().any():
        raise ValueError(
            "Golden set contains missing human_intent labels."
        )

    # ---------------------------------------------------------
    # Baseline evaluation
    # ---------------------------------------------------------
    #
    # The golden set is currently our reviewed evaluation set.
    #
    # For this trivial baseline, the majority class is computed
    # from the evaluated labels only to establish a simple lower
    # bound. Later baselines will use a proper train/test split.
    # ---------------------------------------------------------

    predictions, majority_label = majority_class_predict(
        df,
        df,
        label_column="human_intent",
    )

    y_true = df["human_intent"]

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("MAJORITY CLASS BASELINE")
    print("=" * 60)

    print(f"\nMajority intent: {majority_label}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()