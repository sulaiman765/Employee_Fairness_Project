import os
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Ensure we get the correct project directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Correct paths for model and dataset
model_path = os.path.join(base_dir, "..", "random_forest_model.pkl")
dataset_path = os.path.join(base_dir, "..", "data", "processed_employee_attrition.csv")
results_path = os.path.join(base_dir, "..", "models", "permutation_importance_results.csv")

# Check if the model exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

# Load trained Random Forest model
rf_model = joblib.load(model_path)
print(f"\n✅ Successfully loaded model from: {model_path}")

# Load dataset
df = pd.read_csv(dataset_path)
print(f"\n✅ Successfully loaded dataset from: {dataset_path}")

# Define features and target
X = df.drop(columns=["Attrition"])  # Drop target column
y = df["Attrition"]

# Print dataset details
print("\n🔍 Checking dataset before computing importance:")
print(df.head())  # Show first few rows
print("\n🔍 Summary statistics:")
print(df.describe())  # Show feature distributions

# Compute permutation importance with AUC score instead of accuracy
result = permutation_importance(rf_model, X, y, scoring="roc_auc", n_repeats=10, random_state=42)


# Extract feature importance scores
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
}).sort_values(by="Importance", ascending=False)

# Ensure 'models/' folder exists before saving results
models_dir = os.path.join(base_dir, "..", "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save results to CSV for analysis
importance_df.to_csv(results_path, index=False)
print(f"\n✅ Permutation Importance results saved in: {results_path}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# Print top 10 features
print("\n🔹 Top 10 Most Important Features:")
print(importance_df.head(10))
