import pandas as pd


def filter_brand_interactions(
    interactions,
    brand="AppleSupport",
):
    """
    Filter the interaction dataset to one support brand.
    """

    brand_interactions = interactions[
        interactions["brand"] == brand
    ].copy()

    brand_interactions = brand_interactions.reset_index(drop=True)

    return brand_interactions


def brand_summary(interactions):
    """
    Summarize the selected brand's historical interactions.
    """

    return {
        "interactions": len(interactions),
        "unique_customers": interactions["customer_tweet_id"].nunique(),
        "unique_support_replies": interactions["support_tweet_id"].nunique(),
        "median_response_minutes": interactions[
            "response_time_minutes"
        ].median(),
        "p95_response_minutes": interactions[
            "response_time_minutes"
        ].quantile(0.95),
    }