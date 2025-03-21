# fairness_utils.py
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score
)


def compute_standard_metrics(y_true, y_pred, y_scores=None):
    """
    Compute standard performance metrics.

    Parameters:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_scores: (Optional) Predicted probabilities for the positive class, for ROC-AUC.

    Returns:
        A dictionary with computed metrics.
    """
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1 Score": f1_score,
        "Balanced Accuracy": balanced_accuracy_score
    }
    results = {name: func(y_true, y_pred) for name, func in metrics.items()}
    if y_scores is not None:
        results["ROC-AUC"] = roc_auc_score(y_true, y_scores)
    return results


def compute_fairness(y_true, y_pred, sensitive_features, custom_metrics=None):
    """
    Compute fairness metrics for each sensitive group using Fairlearn's MetricFrame.

    Parameters:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive_features: A dict or DataFrame of sensitive feature(s) (e.g., {"Gender": ..., "OverTime": ...}).
        custom_metrics: (Optional) A dict of additional metrics to compute.

    Returns:
        A tuple of overall metrics and group-specific metrics.
    """
    # Default metrics are the standard performance metrics.
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1 Score": f1_score
    }
    if custom_metrics is not None:
        metrics.update(custom_metrics)

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    overall = metric_frame.overall
    by_group = metric_frame.by_group
    return overall, by_group


# Example for adding a custom fairness metric (optional)
def statistical_parity_difference(y_true, y_pred, sensitive_feature):
    """
    Compute the Statistical Parity Difference for a binary sensitive feature.
    """
    # Convert to a DataFrame for convenience.
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "sensitive": sensitive_feature})
    positive_rate_priv = df.loc[df["sensitive"] == 1, "y_pred"].mean()
    positive_rate_unpriv = df.loc[df["sensitive"] == 0, "y_pred"].mean()
    return positive_rate_unpriv - positive_rate_priv

# You can later integrate the above custom metric into compute_fairness if desired.
