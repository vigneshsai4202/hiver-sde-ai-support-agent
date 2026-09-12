import pandas as pd


def calculate_brand_metrics(df):
    """
    Calculate quantitative metrics for each support brand.

    Metrics:
    - support_tweets:
        Number of outbound/support tweets.

    - customer_tweets_linked:
        Number of unique inbound customer tweets that can be
        directly linked to a support account through
        response_tweet_id.

    These metrics are used only for the first brand-selection
    screen. They do not represent actual resolution rates.
    """

    df = df.copy()

    # Normalize IDs
    df["tweet_id"] = df["tweet_id"].astype(str)

    df["response_tweet_id"] = (
        df["response_tweet_id"]
        .fillna("")
        .astype(str)
    )

    # Outbound/support tweets
    support = df[df["inbound"] == False]

    support_counts = (
        support
        .groupby("author_id")
        .size()
        .rename("support_tweets")
    )

    # Support tweet ID -> brand
    support_lookup = (
        support[
            ["tweet_id", "author_id"]
        ]
        .drop_duplicates("tweet_id")
        .rename(
            columns={
                "tweet_id": "response_id",
                "author_id": "brand"
            }
        )
    )

    # Inbound/customer tweets
    customer = (
        df[df["inbound"] == True][
            ["tweet_id", "response_tweet_id"]
        ]
        .rename(
            columns={
                "tweet_id": "customer_tweet_id"
            }
        )
    )

    # Keep only customer tweets that have responses
    customer = customer[
        customer["response_tweet_id"].str.strip() != ""
    ].copy()

    # Expand comma-separated response IDs
    customer["response_id"] = (
        customer["response_tweet_id"].str.split(",")
    )

    customer = customer.explode("response_id")

    customer["response_id"] = (
        customer["response_id"]
        .astype(str)
        .str.strip()
    )

    customer = customer[
        customer["response_id"] != ""
    ]

    # Link customer tweets -> support brand
    linked = customer.merge(
        support_lookup,
        on="response_id",
        how="inner"
    )

    # Count customer once per brand
    linked = linked[
        ["customer_tweet_id", "brand"]
    ].drop_duplicates()

    customer_counts = (
        linked
        .groupby("brand")
        .size()
        .rename("customer_tweets_linked")
    )

    # Combine metrics
    metrics = pd.concat(
        [
            support_counts,
            customer_counts
        ],
        axis=1
    ).fillna(0)

    metrics["support_tweets"] = (
        metrics["support_tweets"].astype(int)
    )

    metrics["customer_tweets_linked"] = (
        metrics["customer_tweets_linked"].astype(int)
    )

    # Make brand an explicit column
    metrics = (
        metrics
        .sort_values(
            by=[
                "customer_tweets_linked",
                "support_tweets"
            ],
            ascending=False
        )
        .reset_index()
        .rename(
            columns={
                "author_id": "brand",
                "index": "brand"
            }
        )
    )

    return metrics


def save_brand_report(metrics, output_path):
    """Save brand-selection metrics to CSV."""

    metrics.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":
    print("Brand selection module loaded successfully.")
