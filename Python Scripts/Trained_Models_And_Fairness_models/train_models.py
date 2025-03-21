import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Ensure the correct project directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory

# Correct path to dataset
file_path = os.path.join(base_dir, "..", "data", "processed_employee_attrition.csv")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

# Load dataset
df = pd.read_csv(file_path)
print(f"\n✅ Successfully loaded dataset from: {file_path}")

# Check class imbalance
print("\n🔍 Class Distribution in Dataset:")
print(df["Attrition"].value_counts())

# Separate features (X) and target variable (y)
X = df.drop(columns=["Attrition"])  # All columns except the target
y = df["Attrition"]  # Target column (0 = No, 1 = Yes)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 🔍 Check new class distribution after SMOTE
print("\n🔍 Class Distribution After SMOTE:")
print(pd.Series(y_train).value_counts())

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

# Show classification report
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# 🔍 Debugging: Print sample predictions vs. actual labels
print("\n🔍 Sample Predictions (first 10):")
print(y_pred[:10])

print("\n🔍 Sample Actual Labels (first 10):")
print(y_test[:10])

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

# Show classification report
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# 🔍 Debugging Code to Check the File Path Issue
print(f"\n🔍 Current Working Directory: {os.getcwd()}")
print("\n🔍 Sample y_test:")
print(y_test.head())
print("\n🔍 Sample y_pred:")
print(y_pred[:5])

# Ensure 'data/' folder exists before saving files
data_dir = os.path.join(base_dir, "..", "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Save test labels and predictions for fairness analysis
y_test_path = os.path.join(data_dir, "y_test.csv")
y_pred_path = os.path.join(data_dir, "y_pred.csv")

pd.DataFrame({"Attrition": y_test}).to_csv(y_test_path, index=False)
pd.DataFrame({"Predicted": y_pred}).to_csv(y_pred_path, index=False)
print(f"\n✅ Predictions saved in: {y_test_path} and {y_pred_path}")

# Save the trained Random Forest model
model_path = os.path.join(base_dir, "..", "random_forest_model.pkl")
joblib.dump(clf, model_path)
print(f"\n✅ Random Forest model saved as '{model_path}'")










