def audit(df):
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "duplicate_tweet_ids": int(df["tweet_id"].duplicated().sum()),
        "missing_text": int(df["text"].isna().sum()),
        "inbound_rows": int(df["inbound"].sum()),
        "outbound_rows": int((~df["inbound"]).sum()),
        "unique_authors": int(df["author_id"].nunique()),
    }

def brand_counts(df):
    return (df.loc[~df["inbound"]]
            .groupby("author_id").size()
            .sort_values(ascending=False)
            .rename("support_tweets").reset_index())
