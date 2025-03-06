import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed dataset
df = pd.read_csv("data/processed_employee_attrition.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=["Attrition"])  # All columns except the target
y = df["Attrition"]  # Target column (0 = No, 1 = Yes)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

# 🔍 Debugging Code to Check the File Path Issue
import os

# Print the current working directory to check where the script is running
print(f"\n🔍 Current Working Directory: {os.getcwd()}")

# Print a preview of y_test and y_pred before saving
print("\n🔍 Sample y_test:")
print(y_test.head())

print("\n🔍 Sample y_pred:")
print(y_pred[:5])

# Save test labels and predictions for fairness analysis
pd.DataFrame({"Attrition": y_test}).to_csv("data/y_test.csv", index=False)
pd.DataFrame({"Predicted": y_pred}).to_csv("data/y_pred.csv", index=False)
print("\n✅ Predictions saved for fairness analysis.")

