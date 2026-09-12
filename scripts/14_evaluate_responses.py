import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.classifier.classify import classify_message
from src.retrieval.retrieve import HistoricalRetriever
from src.response.generate import generate_response


GOLDEN_PATH = "data/golden/golden_set_v1.csv"
INTERACTIONS_PATH = "data/processed/applesupport_interactions.csv"

OUTPUT_DIR = Path("data/processed/response_evaluation")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = OUTPUT_DIR / "response_predictions.csv"


def main():

    print("=" * 60)
    print("RESPONSE GENERATION EVALUATION")
    print("=" * 60)

    df = pd.read_csv(
        GOLDEN_PATH,
        encoding="latin1",
    )

    retriever = HistoricalRetriever(
        INTERACTIONS_PATH
    )

    # Resume if a previous run exists.
    if OUTPUT_PATH.exists():

        results = pd.read_csv(
            OUTPUT_PATH,
            encoding="utf-8",
        )

        completed = set(
            results["golden_id"].astype(str)
        )

        print(
            f"Resuming: {len(completed)} examples already completed."
        )

    else:

        results = pd.DataFrame()

        completed = set()

    for index, row in df.iterrows():

        golden_id = str(row["golden_id"])

        if golden_id in completed:
            continue

        print(
            f"[{index + 1}/{len(df)}] {golden_id} ..."
        )

        customer_text = str(
            row["text"]
        )

        previous_context = str(
            row["previous_context"]
        )

        try:

            # 1. Classify
            classification = classify_message(
                customer_text=customer_text,
                previous_context=previous_context,
            )

            # 2. Retrieve
            retrieval_results = retriever.retrieve(
                customer_text,
                top_k=5,
            )

            # 3. Generate
            generated_response = generate_response(
                customer_text=customer_text,
                intent=classification["intent"],
                historical_examples=retrieval_results,
            )

            result = {
                "golden_id": golden_id,
                "customer_text": customer_text,
                "previous_context": previous_context,
                "human_intent": row["human_intent"],
                "human_escalate": row["human_escalate"],
                "human_resolution_note": row[
                    "human_resolution_note"
                ],
                "predicted_intent": classification[
                    "intent"
                ],
                "intent_confidence": classification[
                    "confidence"
                ],
                "generated_response": generated_response,
                "retrieval_count": len(
                    retrieval_results
                ),
                "best_similarity": (
                    max(
                        x["similarity"]
                        for x in retrieval_results
                    )
                    if retrieval_results
                    else 0.0
                ),
                "status": "success",
            }

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

            result = {
                "golden_id": golden_id,
                "customer_text": customer_text,
                "previous_context": previous_context,
                "human_intent": row["human_intent"],
                "human_escalate": row["human_escalate"],
                "human_resolution_note": row[
                    "human_resolution_note"
                ],
                "predicted_intent": "",
                "intent_confidence": 0.0,
                "generated_response": "",
                "retrieval_count": 0,
                "best_similarity": 0.0,
                "status": "error",
            }

        results = pd.concat(
            [
                results,
                pd.DataFrame([result]),
            ],
            ignore_index=True,
        )

        results.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8",
        )

        # Keep within Groq free-tier request limits.
        time.sleep(2.0)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Examples: {len(results)}"
    )

    print(
        "Successful:",
        (results["status"] == "success").sum(),
    )

    print(
        "Errors:",
        (results["status"] == "error").sum(),
    )


if __name__ == "__main__":
    main()