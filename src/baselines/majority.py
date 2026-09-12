import pandas as pd


def majority_class_predict(
    train_df,
    test_df,
    label_column="human_intent",
):
    """
    Predict the most frequent class from the training data
    for every example in the test data.
    """

    majority_label = (
        train_df[label_column]
        .value_counts()
        .idxmax()
    )

    predictions = pd.Series(
        majority_label,
        index=test_df.index,
        name="prediction",
    )

    return predictions, majority_label