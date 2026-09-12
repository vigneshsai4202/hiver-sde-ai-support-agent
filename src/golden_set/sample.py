def sample_candidate(df, n=200, random_state=42):
    # One example per conversation prevents leakage.
    candidates = df[df["inbound"]].drop_duplicates("conversation_id")
    return candidates.sample(min(n, len(candidates)), random_state=random_state)
