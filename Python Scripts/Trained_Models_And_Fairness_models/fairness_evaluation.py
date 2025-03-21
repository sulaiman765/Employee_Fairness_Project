import os
import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score

# ✅ Ensure 'results' directory exists
os.makedirs("results", exist_ok=True)

# ✅ Load test labels
y_test = pd.read_csv("data/y_test.csv")["Attrition"]

# ✅ Load predictions for both FCNN models
y_pred_fcnn_baseline = pd.read_csv("data/y_pred_fcnn_baseline.csv")["Predicted"]
y_pred_fcnn_improved = pd.read_csv("data/y_pred_fcnn_improved.csv")["Predicted"]

# ✅ Load dataset to extract sensitive features
df = pd.read_csv("data/processed_employee_attrition.csv")
sensitive_features = {
    "Gender": df.loc[y_test.index, "Gender"],
    "OverTime": df.loc[y_test.index, "OverTime"],
    "BusinessTravel": df.loc[y_test.index, "BusinessTravel"],
}

# ✅ Define fairness metrics (Fix: Prevent recall errors)
metrics = {
    "Accuracy": accuracy_score,
    "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0)
}

# ✅ Compute fairness for both FCNN models
fairness_baseline = MetricFrame(
    metrics=metrics, y_true=y_test, y_pred=y_pred_fcnn_baseline, sensitive_features=sensitive_features
)
fairness_improved = MetricFrame(
    metrics=metrics, y_true=y_test, y_pred=y_pred_fcnn_improved, sensitive_features=sensitive_features
)

# ✅ Print fairness comparison
print("\n🔹 Fairness Comparison:")
print("\n--- Baseline FCNN ---")
print(fairness_baseline.by_group)

print("\n--- Improved FCNN ---")
print(fairness_improved.by_group)

# ✅ Save fairness comparison
fairness_comparison = pd.concat(
    [fairness_baseline.by_group.add_suffix("_Baseline"), fairness_improved.by_group.add_suffix("_Improved")],
    axis=1
)
fairness_comparison.to_csv("results/fairness_comparison.csv")
print("\n✅ Fairness comparison saved as 'results/fairness_comparison.csv'")

