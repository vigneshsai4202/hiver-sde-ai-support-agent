import pandas as pd


ID_COLUMNS = [
    "tweet_id",
    "author_id",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def load_tweets(path):
    """
    Load the customer-support Twitter dataset.

    IDs are loaded as strings because they are identifiers, not numbers.
    Latin-1 is used because that is the encoding of the Kaggle CSV.
    """

    dtype = {
        column: "string"
        for column in ID_COLUMNS
    }

    return pd.read_csv(
        path,
        encoding="latin1",
        dtype=dtype,
        low_memory=False,
    )