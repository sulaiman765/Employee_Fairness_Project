import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score

# ✅ Ensure correct file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed_employee_attrition.csv")

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)

# Prepare data (NO SMOTE APPLIED)
X = df.drop(columns=["Attrition"]).values
y = df["Attrition"].values

# Train-test split (NO Oversampling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

# ✅ Define Basic FCNN Model (NO Fairness Improvements)
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
        x = self.fc3(x)  # Raw logits (No softmax for SHAP)
        return x

# Initialize model
input_size = X_train.shape[1]
model = FCNN(input_size).to(device)

# ✅ NO CLASS WEIGHTING IN LOSS FUNCTION (Baseline)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model (Baseline FCNN)
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ✅ Evaluate model
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()

accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f"\n✅ Baseline FCNN Accuracy: {accuracy:.4f}")
print(f"✅ Baseline FCNN Recall: {recall:.4f}")

# ✅ Save test labels & predictions for fairness evaluation
os.makedirs(os.path.join(BASE_DIR, "../data"), exist_ok=True)  # Ensure 'data' directory exists

pd.DataFrame({"Attrition": y_test}).to_csv(os.path.join(BASE_DIR, "../data/y_test_baseline.csv"), index=False)
pd.DataFrame({"Predicted": predictions}).to_csv(os.path.join(BASE_DIR, "../data/y_pred_fcnn_baseline.csv"), index=False)

print("\n✅ Baseline FCNN predictions saved as 'data/y_pred_fcnn_baseline.csv'")
