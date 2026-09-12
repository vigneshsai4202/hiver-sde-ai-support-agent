import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.classifier.classify import (
    create_client,
    classify_intent,
)


INPUT_PATH = "data/golden/golden_set_v1.csv"

OUTPUT_DIR = Path("data/processed/ai_evaluation")

PREDICTIONS_PATH = (
    OUTPUT_DIR / "groq_predictions.csv"
)

CONFUSION_PATH = (
    OUTPUT_DIR / "groq_confusion_matrix.csv"
)

METRICS_PATH = (
    OUTPUT_DIR / "groq_metrics.csv"
)


MODEL = "openai/gpt-oss-20b"

# Groq free-tier friendly pacing.
# 200 requests at ~2.2 seconds/request ≈ 7-8 minutes.
REQUEST_DELAY_SECONDS = 2.0


def main():

    load_dotenv()

    print("Loading golden dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Golden examples: {len(df)}")

    required_columns = [
        "golden_id",
        "text",
        "previous_context",
        "human_intent",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    client = create_client()

    predictions = []

    print()
    print("=" * 60)
    print("GROQ AI CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Examples: {len(df)}")
    print()

    for index, row in df.iterrows():

        golden_id = row["golden_id"]

        text = (
            ""
            if pd.isna(row["text"])
            else str(row["text"])
        )

        context = (
            ""
            if pd.isna(row["previous_context"])
            else str(row["previous_context"])
        )

        print(
            f"[{index + 1:03d}/{len(df)}] "
            f"{golden_id}",
            end=" ... ",
            flush=True
        )

        try:

            result = classify_intent(
                client=client,
                customer_text=text,
                previous_context=context,
                model=MODEL,
            )

            prediction = {
                "golden_id": golden_id,
                "expected_intent": row["human_intent"],
                "predicted_intent": result["intent"],
                "confidence": result["confidence"],
                "reason": "",
                "correct": (
                    result["intent"]
                    == row["human_intent"]
                ),
                "error": "",
            }

            print(
                f"{result['intent']} "
                f"(confidence={result['confidence']:.2f})"
            )

        except Exception as exc:

            prediction = {
                "golden_id": golden_id,
                "expected_intent": row["human_intent"],
                "predicted_intent": "ERROR",
                "confidence": 0.0,
                "reason": "",
                "correct": False,
                "error": repr(exc),
            }

            print(f"ERROR: {exc}")

        predictions.append(prediction)

        # Avoid hitting RPM limits.
        if index < len(df) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

        # Save progress after every example.
        pd.DataFrame(predictions).to_csv(
            PREDICTIONS_PATH,
            index=False,
        )

    results = pd.DataFrame(predictions)

    valid = results[
        results["predicted_intent"] != "ERROR"
    ].copy()

    y_true = valid["expected_intent"]
    y_pred = valid["predicted_intent"]

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 60)
    print("FINAL GROQ RESULTS")
    print("=" * 60)

    print(
        f"Successful predictions: {len(valid)}/{len(results)}"
    )

    print(
        f"Accuracy:        {accuracy:.4f}"
    )

    print(
        f"Macro F1:        {macro_f1:.4f}"
    )

    print(
        f"Weighted F1:     {weighted_f1:.4f}"
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    confusion_df.to_csv(
        CONFUSION_PATH
    )

    # Overall metrics
    metrics_df = pd.DataFrame(
        [
            {
                "model": MODEL,
                "examples": len(results),
                "successful_predictions": len(valid),
                "errors": len(results) - len(valid),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        ]
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    # Save only incorrect predictions
    failures_path = (
        OUTPUT_DIR / "groq_failures.csv"
    )

    failures = results[
        results["correct"] == False
    ]

    failures.to_csv(
        failures_path,
        index=False,
    )

    print()
    print("Saved:")
    print(f"  {PREDICTIONS_PATH}")
    print(f"  {METRICS_PATH}")
    print(f"  {CONFUSION_PATH}")
    print(f"  {failures_path}")

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()