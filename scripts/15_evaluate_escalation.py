import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# Allow imports when running from the repository root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


INPUT_FILE = ROOT / "data" / "processed" / "response_evaluation" / "response_predictions.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "escalation_evaluation"


AUTO_THRESHOLD = 0.85
RETRIEVAL_THRESHOLD = 0.30
HIGH_RISK_INTENTS = {"PAYMENTS_PURCHASES"}


def decide_escalation(row):
    """
    Apply the project's deterministic escalation policy.

    ESCALATE when:
    - intent confidence is below threshold
    - no historical retrieval evidence exists
    - best historical similarity is weak
    - payment/purchase issue is detected

    Otherwise AUTO.
    """

    confidence = float(row["intent_confidence"])
    similarity = float(row["best_similarity"])
    intent = str(row["predicted_intent"])

    if confidence < AUTO_THRESHOLD:
        return "ESCALATE"

    if similarity < RETRIEVAL_THRESHOLD:
        return "ESCALATE"

    if intent in HIGH_RISK_INTENTS:
        return "ESCALATE"

    return "AUTO"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    # Only evaluate successful pipeline runs.
    df = df[df["status"].astype(str).str.lower() == "success"].copy()

    if df.empty:
        raise ValueError("No successful predictions available for escalation evaluation.")

    df["predicted_escalation"] = df.apply(decide_escalation, axis=1)

    y_true = df["human_escalate"].astype(bool)
    y_pred = df["predicted_escalation"].eq("ESCALATE")

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=True,
        zero_division=0,
    )

    print("\n=== Escalation Evaluation ===")
    print(f"Examples evaluated: {len(df)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[False, True],
            target_names=["AUTO", "ESCALATE"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    print("=== Confusion Matrix ===")
    print("                 Pred AUTO   Pred ESCALATE")
    print(f"Human AUTO       {cm[0,0]:10d}   {cm[0,1]:14d}")
    print(f"Human ESCALATE   {cm[1,0]:10d}   {cm[1,1]:14d}")

    # Save row-level evaluation.
    predictions_file = OUTPUT_DIR / "escalation_predictions.csv"
    df.to_csv(predictions_file, index=False)

    # Save metrics.
    metrics = pd.DataFrame(
        [
            {
                "examples_evaluated": len(df),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auto_threshold": AUTO_THRESHOLD,
                "retrieval_threshold": RETRIEVAL_THRESHOLD,
            }
        ]
    )

    metrics_file = OUTPUT_DIR / "escalation_metrics.csv"
    metrics.to_csv(metrics_file, index=False)

    # Save confusion matrix.
    cm_df = pd.DataFrame(
        cm,
        index=["human_AUTO", "human_ESCALATE"],
        columns=["pred_AUTO", "pred_ESCALATE"],
    )

    cm_file = OUTPUT_DIR / "escalation_confusion_matrix.csv"
    cm_df.to_csv(cm_file)

    # Save mistakes for failure analysis.
    errors = df[y_true != y_pred].copy()

    errors_file = OUTPUT_DIR / "escalation_errors.csv"
    errors.to_csv(errors_file, index=False)

    print("\nSaved:")
    print(predictions_file)
    print(metrics_file)
    print(cm_file)
    print(errors_file)


if __name__ == "__main__":
    main()