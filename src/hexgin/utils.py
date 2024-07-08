"""A module containing utility functions for the project."""

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def sklearn_classification_report_to_pandas(y_true, y_pred) -> pd.DataFrame:
    """
    Convert a sklearn classification report to a pandas DataFrame.
    The code is based on https://stackoverflow.com/a/50091428
    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred: array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    Returns
    -------
    df_classification_report: pd.DataFrame
        pandas DataFrame containing the classification report
    """
    metrics_summary = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)

    avg = list(
        precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="weighted"
        )
    )

    metrics_sum_index = ["precision", "recall", "f1-score", "support"]
    class_report_df = pd.DataFrame(list(metrics_summary), index=metrics_sum_index)

    support = class_report_df.loc["support"]
    total = support.sum()
    avg[-1] = total

    class_report_df["avg / total"] = avg

    return class_report_df.T.reset_index().rename(columns={"index": "class"})
