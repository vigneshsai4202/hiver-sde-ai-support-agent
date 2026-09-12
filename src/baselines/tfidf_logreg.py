import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_tfidf_logreg(
    train_df,
    text_column="text",
    label_column="human_intent",
):
    """
    Train a TF-IDF + Logistic Regression classifier.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(
        train_df[text_column].fillna("")
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        train_df[label_column],
    )

    return vectorizer, model


def predict_tfidf_logreg(
    vectorizer,
    model,
    test_df,
    text_column="text",
):
    """
    Predict intents for new examples.
    """

    X_test = vectorizer.transform(
        test_df[text_column].fillna("")
    )

    predictions = model.predict(X_test)

    return pd.Series(
        predictions,
        index=test_df.index,
        name="prediction",
    )