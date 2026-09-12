import os
import re
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PATHS / CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "judge_evaluation"
    / "judge_human_review.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "judge_evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "llm_judge_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "judge_metrics.csv"
)

AGREEMENT_FILE = (
    OUTPUT_DIR
    / "judge_human_agreement.csv"
)

MODEL = "openai/gpt-oss-20b"

# Keep this reasonably small because Groq daily token
# usage is limited.
MAX_COMPLETION_TOKENS = 256

SLEEP_SECONDS = 2


# ============================================================
# JUDGE PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an evaluator for a customer-support AI system.

Evaluate the generated customer-support response using ONLY:

1. the customer's message
2. previous conversation context
3. the generated response

Do not assume facts that are not present in the supplied information.

Evaluate these dimensions.

RELEVANCE:
1 = does not address the customer's issue
2 = mostly off-topic
3 = partially addresses the issue
4 = clearly relevant
5 = directly addresses the issue

GROUNDEDNESS:
1 = mostly unsupported or invented
2 = several unsupported claims
3 = partially supported
4 = mostly grounded
5 = fully grounded with no meaningful unsupported claims

HELPFULNESS:
1 = not useful
2 = barely useful
3 = somewhat useful
4 = useful next step
5 = clear, actionable and customer-appropriate

UNSUPPORTED_CLAIMS:
0 = no meaningful unsupported claim
1 = at least one meaningful unsupported claim

OVERALL:
1 = poor
2 = weak
3 = acceptable
4 = good
5 = excellent

Return ONLY one compact JSON object.

Example:
{"relevance":4,"groundedness":4,"helpfulness":4,"unsupported_claims":0,"overall":4}

Do not provide explanations.
Do not use markdown.
"""


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text):
    """
    Extract the first JSON object from model output.
    """

    if not text:
        return None

    text = text.strip()

    # First try exact JSON.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find JSON embedded in the response.
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        return None

    try:
        return json.loads(
            match.group(0)
        )
    except Exception:
        return None


# ============================================================
# VALIDATION
# ============================================================

def validate_result(result):
    """
    Validate judge JSON structure and score ranges.
    """

    if not isinstance(result, dict):
        return False

    required = [
        "relevance",
        "groundedness",
        "helpfulness",
        "unsupported_claims",
        "overall",
    ]

    if not all(
        key in result
        for key in required
    ):
        return False

    try:
        relevance = int(
            result["relevance"]
        )

        groundedness = int(
            result["groundedness"]
        )

        helpfulness = int(
            result["helpfulness"]
        )

        unsupported_claims = int(
            result["unsupported_claims"]
        )

        overall = int(
            result["overall"]
        )

    except Exception:
        return False

    return (
        1 <= relevance <= 5
        and 1 <= groundedness <= 5
        and 1 <= helpfulness <= 5
        and unsupported_claims in [0, 1]
        and 1 <= overall <= 5
    )


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(row):

    customer = str(
        row.get(
            "customer_text",
            ""
        )
    )

    context = str(
        row.get(
            "previous_context",
            ""
        )
    )

    response = str(
        row.get(
            "generated_response",
            ""
        )
    )

    if context.lower() == "nan":
        context = (
            "(No previous context provided.)"
        )

    return f"""
CUSTOMER MESSAGE:
{customer}

PREVIOUS CONTEXT:
{context}

GENERATED SUPPORT RESPONSE:
{response}

Evaluate the generated response according to the rubric.

Return only the compact JSON object.
"""


# ============================================================
# SINGLE JUDGE CALL
# ============================================================

def judge_one(client, row):

    prompt = build_prompt(row)

    completion = (
        client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0,

            max_completion_tokens=MAX_COMPLETION_TOKENS,

            reasoning_effort="low",

            include_reasoning=False,
        )
    )

    content = (
        completion
        .choices[0]
        .message
        .content
    )

    result = extract_json(
        content
    )

    if not validate_result(
        result
    ):
        raise ValueError(
            f"Invalid judge output: {content}"
        )

    return result


# ============================================================
# AGREEMENT CALCULATION
# ============================================================

def calculate_agreement(df):

    metrics = []

    score_columns = [
        (
            "judge_relevance",
            "human_relevance",
        ),
        (
            "judge_groundedness",
            "human_groundedness",
        ),
        (
            "judge_helpfulness",
            "human_helpfulness",
        ),
        (
            "judge_overall",
            "human_overall",
        ),
    ]

    for judge_col, human_col in score_columns:

        if judge_col not in df.columns:
            print(
                f"Skipping missing column: "
                f"{judge_col}"
            )
            continue

        if human_col not in df.columns:
            print(
                f"Skipping missing column: "
                f"{human_col}"
            )
            continue

        judge = pd.to_numeric(
            df[judge_col],
            errors="coerce",
        )

        human = pd.to_numeric(
            df[human_col],
            errors="coerce",
        )

        valid = (
            judge.notna()
            & human.notna()
        )

        judge = judge[valid]
        human = human[valid]

        if len(judge) == 0:
            continue

        absolute_difference = (
            judge - human
        ).abs()

        exact_agreement = (
            absolute_difference == 0
        ).mean()

        within_1_agreement = (
            absolute_difference <= 1
        ).mean()

        metrics.append(
            {
                "dimension": (
                    judge_col
                    .replace(
                        "judge_",
                        ""
                    )
                ),

                "n": len(judge),

                "mean_human": (
                    human.mean()
                ),

                "mean_judge": (
                    judge.mean()
                ),

                "mean_absolute_difference": (
                    absolute_difference.mean()
                ),

                "exact_agreement": (
                    exact_agreement
                ),

                "within_1_agreement": (
                    within_1_agreement
                ),
            }
        )

    return pd.DataFrame(
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load environment variables
    # --------------------------------------------------------

    load_dotenv(
        ROOT / ".env"
    )

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY not found in .env"
        )


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing input file:\n"
            f"{INPUT_FILE}"
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load review dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "=== LLM-as-Judge Evaluation ==="
    )

    print(
        f"Examples to judge: {len(df)}"
    )

    print(
        f"Model: {MODEL}"
    )

    print()


    # --------------------------------------------------------
    # Groq client
    # --------------------------------------------------------

    client = Groq(
        api_key=api_key
    )


    # --------------------------------------------------------
    # Resume previous results
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():

        existing = pd.read_csv(
            OUTPUT_FILE
        )

        # Remove duplicate IDs from
        # previous interrupted runs.
        if "golden_id" in existing.columns:

            existing = (
                existing
                .drop_duplicates(
                    subset=[
                        "golden_id"
                    ],
                    keep="last",
                )
            )

        # Only successful rows are considered
        # completed.
        if (
            "golden_id" in existing.columns
            and "judge_status" in existing.columns
        ):

            completed = set(
                existing.loc[
                    existing[
                        "judge_status"
                    ]
                    .astype(str)
                    .str.lower()
                    .eq("success"),

                    "golden_id",
                ]
                .astype(str)
            )

        else:

            completed = set()


        results = existing.to_dict(
            orient="records"
        )

        print(
            "Existing successful judge results: "
            f"{len(completed)}"
        )

    else:

        completed = set()
        results = []

        print(
            "Existing successful judge results: 0"
        )


    # --------------------------------------------------------
    # Process examples
    # --------------------------------------------------------

    for index, row in df.iterrows():

        golden_id = str(
            row["golden_id"]
        )


        # Skip only successful examples.
        if golden_id in completed:
            continue


        print(
            f"[{index + 1}/{len(df)}] "
            f"Judging {golden_id}..."
        )


        base = row.to_dict()


        # ----------------------------------------------------
        # Call judge
        # ----------------------------------------------------

        try:

            result = judge_one(
                client,
                row
            )


            base.update(
                {
                    "judge_relevance":
                        result["relevance"],

                    "judge_groundedness":
                        result["groundedness"],

                    "judge_helpfulness":
                        result["helpfulness"],

                    "judge_unsupported_claims":
                        result[
                            "unsupported_claims"
                        ],

                    "judge_overall":
                        result["overall"],

                    "judge_status":
                        "success",

                    "judge_error":
                        "",
                }
            )


            print(
                f"  Overall: "
                f"{result['overall']}/5 | "
                f"Groundedness: "
                f"{result['groundedness']}/5"
            )


        # ----------------------------------------------------
        # Error handling
        # ----------------------------------------------------

        except Exception as exc:

            base.update(
                {
                    "judge_relevance": None,

                    "judge_groundedness": None,

                    "judge_helpfulness": None,

                    "judge_unsupported_claims":
                        None,

                    "judge_overall": None,

                    "judge_status":
                        "error",

                    "judge_error":
                        str(exc),
                }
            )


            print(
                f"  ERROR: {exc}"
            )


        # ----------------------------------------------------
        # Replace previous result for this ID
        # ----------------------------------------------------

        results = [
            r
            for r in results
            if str(
                r.get("golden_id")
            ) != golden_id
        ]

        results.append(
            base
        )


        # ----------------------------------------------------
        # Save after every example
        # ----------------------------------------------------

        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_FILE,
            index=False
        )


        # ----------------------------------------------------
        # Delay
        # ----------------------------------------------------

        try:

            time.sleep(
                SLEEP_SECONDS
            )

        except KeyboardInterrupt:

            print(
                "\nInterrupted by user."
            )

            print(
                "Existing results have been saved."
            )

            break


    # ========================================================
    # FINAL DATASET
    # ========================================================

    final_df = pd.DataFrame(
        results
    )


    # Remove duplicates one final time.
    if "golden_id" in final_df.columns:

        final_df = (
            final_df
            .drop_duplicates(
                subset=[
                    "golden_id"
                ],
                keep="last",
            )
        )


    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # SUCCESS / FAILURE COUNTS
    # ========================================================

    successful = final_df[
        final_df[
            "judge_status"
        ]
        .astype(str)
        .str.lower()
        .eq("success")
    ].copy()


    failed = final_df[
        ~final_df[
            "judge_status"
        ]
        .astype(str)
        .str.lower()
        .eq("success")
    ].copy()


    print()

    print(
        "Successful judge evaluations: "
        f"{len(successful)}"
    )

    print(
        "Failed judge evaluations: "
        f"{len(failed)}"
    )


    # ========================================================
    # CALCULATE AGREEMENT
    # ========================================================

    if len(successful) == 0:

        print()
        print(
            "No successful judge evaluations."
        )

        print(
            "Agreement metrics cannot be calculated."
        )

        return


    agreement = calculate_agreement(
        successful
    )


    # --------------------------------------------------------
    # Save agreement
    # --------------------------------------------------------

    agreement.to_csv(
        AGREEMENT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    agreement.to_csv(
        METRICS_FILE,
        index=False
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        "=== Judge-Human Agreement ==="
    )

    if len(agreement) > 0:

        print(
            agreement.to_string(
                index=False
            )
        )

    else:

        print(
            "No agreement metrics available."
        )


    # ========================================================
    # FILES
    # ========================================================

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        METRICS_FILE
    )

    print(
        AGREEMENT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()