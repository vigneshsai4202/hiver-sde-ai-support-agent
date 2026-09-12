from src.data.load import load_tweets
from src.data.audit import audit, brand_counts

df = load_tweets("data/raw/tweets.csv")
print(audit(df))
print(brand_counts(df).head(30).to_string(index=False))
