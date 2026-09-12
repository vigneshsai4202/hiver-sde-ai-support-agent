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
from sklearn.model_selection import train_test_split

from src.baselines.tfidf_logreg import (
    train_tfidf_logreg,
    predict_tfidf_logreg,
)


GOLDEN_PATH = "data/golden/golden_set_v1.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():

    print("Loading golden dataset...")

    df = pd.read_csv(
        GOLDEN_PATH,
        encoding="utf-8",
    )

    print(f"Golden examples: {len(df):,}")

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    if df["human_intent"].isna().any():
        raise ValueError(
            "Golden set contains missing human_intent labels."
        )

    # ---------------------------------------------------------
    # Stratified train/test split
    # ---------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["human_intent"],
    )

    print(f"Training examples: {len(train_df):,}")
    print(f"Test examples:     {len(test_df):,}")

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\nTraining TF-IDF + Logistic Regression...")

    vectorizer, model = train_tfidf_logreg(
        train_df,
        text_column="text",
        label_column="human_intent",
    )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    predictions = predict_tfidf_logreg(
        vectorizer,
        model,
        test_df,
        text_column="text",
    )

    y_true = test_df["human_intent"]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

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
    print("TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    print(f"\nAccuracy:        {accuracy:.4f}")
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