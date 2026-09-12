import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "response_evaluation"
    / "response_predictions.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "judge_evaluation"
OUTPUT_FILE = OUTPUT_DIR / "judge_human_review.csv"

SAMPLE_SIZE = 25
RANDOM_STATE = 42


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    # Keep only successful response generations.
    df = df[
        df["status"].astype(str).str.lower().eq("success")
    ].copy()

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(df)} successful responses available; "
            f"need at least {SAMPLE_SIZE}."
        )

    # Stratify approximately by human escalation label so both
    # AUTO and ESCALATE examples are represented.
    auto = df[df["human_escalate"].astype(bool) == False]
    escalate = df[df["human_escalate"].astype(bool) == True]

    n_escalate = min(10, len(escalate))
    n_auto = SAMPLE_SIZE - n_escalate

    sample_escalate = escalate.sample(
        n=n_escalate,
        random_state=RANDOM_STATE
    )

    sample_auto = auto.sample(
        n=n_auto,
        random_state=RANDOM_STATE
    )

    sample = pd.concat(
        [sample_auto, sample_escalate],
        ignore_index=True
    )

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Create human-review fields.
    sample["human_relevance"] = ""
    sample["human_groundedness"] = ""
    sample["human_helpfulness"] = ""
    sample["human_unsupported_claims"] = ""
    sample["human_overall"] = ""
    sample["human_notes"] = ""

    # Keep the review file focused.
    columns = [
        "golden_id",
        "customer_text",
        "previous_context",
        "human_intent",
        "human_escalate",
        "generated_response",
        "retrieval_count",
        "best_similarity",
        "human_relevance",
        "human_groundedness",
        "human_helpfulness",
        "human_unsupported_claims",
        "human_overall",
        "human_notes",
    ]

    sample[columns].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=== Judge Calibration Set ===")
    print(f"Successful responses available: {len(df)}")
    print(f"Review examples selected: {len(sample)}")
    print(f"Human AUTO examples: {n_auto}")
    print(f"Human ESCALATE examples: {n_escalate}")

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()