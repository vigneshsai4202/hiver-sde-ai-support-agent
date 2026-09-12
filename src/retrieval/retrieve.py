import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalRetriever:
    """
    Retrieve historically similar AppleSupport interactions.

    Retrieval is performed against historical customer messages.
    The corresponding AppleSupport response is returned as handling evidence.
    """

    def __init__(
        self,
        interactions_path,
        max_features=20000,
        ngram_range=(1, 2),
    ):
        self.interactions_path = interactions_path

        self.df = pd.read_csv(
            interactions_path,
            encoding="latin1",
        )

        required_columns = {
            "customer_text",
            "support_text",
            "interaction_id",
            "customer_tweet_id",
            "support_tweet_id",
            "brand",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        self.df["customer_text"] = (
            self.df["customer_text"]
            .fillna("")
            .astype(str)
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_text"]
        )

    @staticmethod
    def _clean_query(text):
        """Remove obvious Twitter noise while preserving issue wording."""

        text = "" if text is None else str(text)

        # Remove URLs
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Remove @mentions
        text = re.sub(
            r"@\w+",
            " ",
            text,
        )

        # Normalize whitespace
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    def retrieve(
        self,
        customer_text,
        top_k=5,
    ):
        """
        Return top-k historically similar interactions.
        """

        query = self._clean_query(customer_text)

        if not query:
            return []

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).ravel()

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for rank, idx in enumerate(top_indices, start=1):

            row = self.df.iloc[idx]

            results.append(
                {
                    "rank": rank,
                    "similarity": float(scores[idx]),
                    "interaction_id": str(
                        row["interaction_id"]
                    ),
                    "customer_tweet_id": str(
                        row["customer_tweet_id"]
                    ),
                    "support_tweet_id": str(
                        row["support_tweet_id"]
                    ),
                    "brand": row["brand"],
                    "customer_text": row["customer_text"],
                    "support_text": row["support_text"],
                }
            )

        return results
