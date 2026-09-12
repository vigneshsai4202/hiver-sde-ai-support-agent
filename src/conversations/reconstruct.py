import pandas as pd


def reconstruct_conversations(df):
    """
    Reconstruct conversation/thread IDs using parent-child relationships.

    Each tweet points to its parent through `in_response_to_tweet_id`.
    Root tweets start conversations. Tweets whose parents are present
    inherit the root conversation ID.

    Missing parent references are treated as conversation boundaries.
    """

    out = df.copy()

    # IDs must remain strings.
    out["tweet_id"] = out["tweet_id"].astype("string")
    out["in_response_to_tweet_id"] = (
        out["in_response_to_tweet_id"].astype("string")
    )

    # Normalize empty relationship values.
    out["in_response_to_tweet_id"] = (
        out["in_response_to_tweet_id"]
        .fillna("")
        .str.strip()
    )

    # Create a direct parent lookup.
    parent_map = dict(
        zip(
            out["tweet_id"],
            out["in_response_to_tweet_id"]
        )
    )

    # Cache computed roots so each tweet is resolved once.
    root_cache = {}

    def find_root(tweet_id):
        if tweet_id in root_cache:
            return root_cache[tweet_id]

        current = tweet_id
        path = []
        visited = set()

        while True:
            if current in root_cache:
                root = root_cache[current]
                break

            if current in visited:
                # Defensive handling for malformed/cyclic data.
                root = current
                break

            visited.add(current)
            path.append(current)

            parent = parent_map.get(current, "")

            # No parent -> this is the root.
            if not parent or parent not in parent_map:
                root = current
                break

            current = parent

        for node in path:
            root_cache[node] = root

        return root

    out["conversation_id"] = [
        find_root(tweet_id)
        for tweet_id in out["tweet_id"]
    ]

    return out


def conversation_summary(df):
    """
    Create one summary row per reconstructed conversation.
    """

    work = df.copy()

    work["created_at"] = pd.to_datetime(
        work["created_at"],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    grouped = work.groupby("conversation_id", sort=False)

    summary = grouped.agg(
        tweet_count=("tweet_id", "count"),
        customer_turns=("inbound", lambda x: int(x.sum())),
        support_turns=("inbound", lambda x: int((~x).sum())),
        start_time=("created_at", "min"),
        end_time=("created_at", "max"),
    ).reset_index()

    summary["duration_minutes"] = (
        summary["end_time"] - summary["start_time"]
    ).dt.total_seconds() / 60.0

    summary["is_two_sided"] = (
        (summary["customer_turns"] > 0)
        & (summary["support_turns"] > 0)
    )

    return summary