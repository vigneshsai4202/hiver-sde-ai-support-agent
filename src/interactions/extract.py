import pandas as pd


REQUIRED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
    "conversation_id",
]


def extract_support_interactions(df):
    """
    Extract direct customer -> support interactions.

    A valid interaction is an outbound/support tweet whose direct
    parent is an inbound/customer tweet.

    We deliberately do NOT treat a reply-connected graph component
    as a single customer conversation.
    """

    work = df[REQUIRED_COLUMNS].copy()

    # Normalize identifiers.
    work["tweet_id"] = work["tweet_id"].astype("string")
    work["author_id"] = work["author_id"].astype("string")

    work["in_response_to_tweet_id"] = (
        work["in_response_to_tweet_id"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    # ---------------------------------------------------------
    # Customer tweets
    # ---------------------------------------------------------

    customer = work[work["inbound"] == True][
        [
            "tweet_id",
            "author_id",
            "created_at",
            "text",
        ]
    ].copy()

    customer = customer.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "created_at": "customer_created_at",
            "text": "customer_text",
        }
    )

    # ---------------------------------------------------------
    # Support tweets
    # ---------------------------------------------------------

    support = work[
        (work["inbound"] == False)
        & (work["in_response_to_tweet_id"] != "")
    ][
        [
            "tweet_id",
            "author_id",
            "created_at",
            "text",
            "in_response_to_tweet_id",
            "conversation_id",
        ]
    ].copy()

    support = support.rename(
        columns={
            "tweet_id": "support_tweet_id",
            "author_id": "brand",
            "created_at": "support_created_at",
            "text": "support_text",
            "in_response_to_tweet_id": "customer_tweet_id",
        }
    )

    # ---------------------------------------------------------
    # Direct customer -> support join
    # ---------------------------------------------------------

    interactions = support.merge(
        customer,
        on="customer_tweet_id",
        how="inner",
    )

    # ---------------------------------------------------------
    # Response time
    # ---------------------------------------------------------

    interactions["customer_created_at"] = pd.to_datetime(
        interactions["customer_created_at"],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    interactions["support_created_at"] = pd.to_datetime(
        interactions["support_created_at"],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    interactions["response_time_minutes"] = (
        interactions["support_created_at"]
        - interactions["customer_created_at"]
    ).dt.total_seconds() / 60.0

    # ---------------------------------------------------------
    # Interaction ID
    # ---------------------------------------------------------

    interactions["interaction_id"] = (
        interactions["customer_tweet_id"].astype(str)
        + "_"
        + interactions["support_tweet_id"].astype(str)
    )

    # ---------------------------------------------------------
    # Final schema
    # ---------------------------------------------------------

    interactions = interactions[
        [
            "interaction_id",
            "customer_tweet_id",
            "support_tweet_id",
            "brand",
            "customer_text",
            "support_text",
            "customer_created_at",
            "support_created_at",
            "response_time_minutes",
            "conversation_id",
        ]
    ]

    return interactions


def interaction_summary(interactions):
    """
    Produce high-level statistics for the extracted interactions.
    """

    return pd.DataFrame(
        {
            "metric": [
                "interactions",
                "unique_customer_tweets",
                "unique_support_tweets",
                "unique_brands",
                "median_response_minutes",
                "p95_response_minutes",
            ],
            "value": [
                len(interactions),
                interactions["customer_tweet_id"].nunique(),
                interactions["support_tweet_id"].nunique(),
                interactions["brand"].nunique(),
                interactions["response_time_minutes"].median(),
                interactions["response_time_minutes"].quantile(0.95),
            ],
        }
    )