import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score

# Load preprocessed dataset
df = pd.read_csv(r"C:\Users\YourUsername\Documents\Employee_Fairness_Project\data\processed_employee_attrition.csv")

# Load test predictions from `train_models.py`
y_test = pd.read_csv("data/y_test.csv")["Attrition"]
y_pred = pd.read_csv("data/y_pred.csv")["Predicted"]

# Select sensitive attributes (Fairness Analysis)
sensitive_features = {
    "Gender": df.loc[y_test.index, "Gender"],  # 0 = Female, 1 = Male
    "OverTime": df.loc[y_test.index, "OverTime"]  # 0 = No, 1 = Yes
}

# Define fairness metrics
metrics = {
    "Accuracy": accuracy_score,
    "Recall": recall_score
}

# Compute fairness metrics
fairness_results = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

# Print fairness metrics
print("\n🔹 Fairness Analysis Results:")
print(fairness_results.by_group)
