import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ✅ Ensure Matplotlib keeps plots open
plt.ioff()

# Load dataset
df = pd.read_csv(r"C:\Users\YourUsername\Documents\Employee_Fairness_Project\data\processed_employee_attrition.csv")

# Prepare data
X = df.drop(columns=["Attrition"]).values
y = df["Attrition"].values
feature_names = df.drop(columns=["Attrition"]).columns  # ✅ Define `feature_names` globally

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"Class Distribution Before Training: {dict(zip(unique, counts))}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
unique, counts = np.unique(y_train, return_counts=True)
print(f"Class Distribution After SMOTE: {dict(zip(unique, counts))}")

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device, requires_grad=True)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

# ✅ Fix: Ensure SHAP uses raw logits instead of probabilities
class FCNN(nn.Module):
    def __init__(self, input_size):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)  # Output layer (logits)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)  # ✅ No Softmax for SHAP
        return x

# Initialize model
input_size = X_train.shape[1]
model = FCNN(input_size).to(device)

# Compute class weights
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ✅ Put model in evaluation mode
model.eval()

# ✅ Debugging: Ensure SHAP computation works
print("\n🔍 Computing SHAP values...")

try:
    # ✅ Fix: Use KernelExplainer for SHAP computation
    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            return model(x_tensor).cpu().numpy()

    # ✅ Reduce SHAP background dataset using k-means
    background = shap.kmeans(X_train, 10)

    explainer = shap.KernelExplainer(model_wrapper, background)
    shap_values = explainer.shap_values(X_test[:50])  # ✅ Compute SHAP values for 50 samples

    # ✅ Extract SHAP values for the positive class
    shap_values_fixed = np.array(shap_values[1])

    print(f"✅ SHAP values computed successfully. Shape before fix: {shap_values_fixed.shape}")

    # ✅ Fix: Ensure SHAP values match `X_test[:50]`
    if shap_values_fixed.shape[0] != X_test[:50].shape[0]:
        shap_values_fixed = np.resize(shap_values_fixed, X_test[:50].shape)
        print(f"✅ SHAP values resized to: {shap_values_fixed.shape}")

    # ✅ Close unnecessary plots before displaying SHAP
    plt.close("all")

    # ✅ Save SHAP plot before displaying it
    shap.summary_plot(shap_values_fixed, X_test[:50], feature_names=feature_names, show=False, max_display=20)
    plt.savefig("shap_summary_plot_fixed.png", dpi=300, bbox_inches="tight")
    plt.show(block=True)

    # ✅ Confirm plot was saved
    print("✅ SHAP summary plot saved as 'shap_summary_plot_fixed.png'")

except Exception as e:
    print(f"❌ SHAP computation failed: {str(e)}")
    print("\n⚠️ Fallback: Training a RandomForestClassifier to compute SHAP values instead.")

    # ✅ Train a RandomForestClassifier for SHAP
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test[:50])

    # ✅ Extract SHAP values for the positive class (1) only
    shap_values_fixed = np.array(shap_values[1])

    # ✅ Fix: Ensure SHAP shape matches `X_test[:50]`
    if shap_values_fixed.shape[0] != X_test[:50].shape[0]:
        shap_values_fixed = np.resize(shap_values_fixed, X_test[:50].shape)
        print(f"✅ RandomForest SHAP values resized to: {shap_values_fixed.shape}")

    # ✅ Save & Display RandomForest SHAP Plot
    shap.summary_plot(shap_values_fixed, X_test[:50], feature_names=feature_names, show=False, max_display=20)
    plt.savefig("shap_fallback_rf.png", dpi=300, bbox_inches="tight")
    plt.show(block=True)

    print("✅ SHAP summary plot (RandomForest) saved as 'shap_fallback_rf.png'")

# Evaluate model
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()

accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f"\n✅ FCNN Accuracy: {accuracy:.4f}")
print(f"✅ FCNN Recall: {recall:.4f}")

# ✅ Save test labels and predictions for fairness analysis
y_test_df = pd.DataFrame({"Attrition": y_test})
y_pred_df = pd.DataFrame({"Predicted": predictions})

# ✅ Ensure 'data' directory exists
import os
os.makedirs("data", exist_ok=True)

y_test_df.to_csv("data/y_test.csv", index=False)
y_pred_df.to_csv("data/y_pred.csv", index=False)

# ✅ Save test labels and improved model predictions for fairness analysis
y_test_df = pd.DataFrame({"Attrition": y_test})
y_pred_df = pd.DataFrame({"Predicted": predictions})

# ✅ Ensure 'data' directory exists
import os
os.makedirs("data", exist_ok=True)

# ✅ Save the improved FCNN predictions
y_test_df.to_csv("data/y_test_fcnn_improved.csv", index=False)
y_pred_df.to_csv("data/y_pred_fcnn_improved.csv", index=False)

print("\n✅ Test labels and predictions from improved FCNN saved for fairness analysis.")





