import pandas as pd
import numpy as np
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed dataset
df = pd.read_csv(r"C:\Users\YourUsername\Documents\Employee_Fairness_Project\data\processed_employee_attrition.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

from imblearn.over_sampling import SMOTE

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"\n🔹 Dataset Balanced with SMOTE: {np.bincount(y_train)}")

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Define both models
logistic_model = LogisticRegression(solver="liblinear", random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train Logistic Regression
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)

# Train Random Forest
random_forest_model.fit(X_train, y_train)
random_forest_pred = random_forest_model.predict(X_test)

# Evaluate Both Models
from sklearn.metrics import accuracy_score, recall_score

logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)

random_forest_accuracy = accuracy_score(y_test, random_forest_pred)
random_forest_recall = recall_score(y_test, random_forest_pred)

print("\n✅ Model Comparison:")
print(f"🔹 Logistic Regression - Accuracy: {logistic_accuracy:.4f}, Recall: {logistic_recall:.4f}")
print(f"🔹 Random Forest - Accuracy: {random_forest_accuracy:.4f}, Recall: {random_forest_recall:.4f}")

# Convert X_train into a DataFrame
X_train_df = pd.DataFrame(X_train, columns=X.columns)

# Ensure sensitive_features is passed as a DataFrame with the correct shape
sensitive_features = X_train_df[["Gender"]]

# Define Fairlearn's bias mitigation method
logistic_fair = ExponentiatedGradient(logistic_model, constraints=EqualizedOdds())
random_forest_fair = ExponentiatedGradient(random_forest_model, constraints=EqualizedOdds())

# Train Both Models with Fairness Constraints
logistic_fair.fit(X_train, y_train, sensitive_features=sensitive_features)
random_forest_fair.fit(X_train, y_train, sensitive_features=sensitive_features)

# Make Predictions
logistic_fair_pred = logistic_fair.predict(X_test)
random_forest_fair_pred = random_forest_fair.predict(X_test)

# Evaluate Fairness-Aware Models
logistic_fair_accuracy = accuracy_score(y_test, logistic_fair_pred)
logistic_fair_recall = recall_score(y_test, logistic_fair_pred)

random_forest_fair_accuracy = accuracy_score(y_test, random_forest_fair_pred)
random_forest_fair_recall = recall_score(y_test, random_forest_fair_pred)

print("\n✅ Fairness-Aware Model Comparison:")
print(f"🔹 Logistic Regression (Fair) - Accuracy: {logistic_fair_accuracy:.4f}, Recall: {logistic_fair_recall:.4f}")
print(f"🔹 Random Forest (Fair) - Accuracy: {random_forest_fair_accuracy:.4f}, Recall: {random_forest_fair_recall:.4f}")

# Make predictions using the fairness-aware Logistic Regression model
y_pred = logistic_fair.predict(X_test)

# Evaluate new fair model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Fair Model Accuracy: {accuracy:.4f}")

# Show classification report
print("\n🔹 Fair Model Classification Report:")
print(classification_report(y_test, y_pred))

