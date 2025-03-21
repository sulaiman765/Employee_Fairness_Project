import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function


# =============================
# 1. Gradient Reversal Layer Definition
# =============================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


# =============================
# 2. Adversarial FCNN Model Definition
# =============================
class AdversarialFCNN(nn.Module):
    def __init__(self, input_size, lambda_adv=1.0):
        super(AdversarialFCNN, self).__init__()
        self.lambda_adv = lambda_adv

        # Shared representation layers
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        # Main prediction head for Attrition (2 classes)
        self.fc_main = nn.Linear(32, 2)

        # Adversary head for predicting Gender (2 classes)
        self.adversary = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        h = self.relu1(self.fc1(x))
        h = self.relu2(self.fc2(h))

        logits_main = self.fc_main(h)
        # Reverse gradients for adversary branch
        h_reversed = grad_reverse(h, self.lambda_adv)
        logits_adv = self.adversary(h_reversed)

        return logits_main, logits_adv


# =============================
# 3. Data Preparation
# =============================
# Load the preprocessed dataset (adjust path as needed)
df = pd.read_csv(r"C:\Users\YourUsername\Documents\Employee_Fairness_Project\data\processed_employee_attrition.csv")

# For adversarial training, remove sensitive attribute "Gender" from main features.
X = df.drop(columns=["Attrition", "Gender"]).values
y_main = df["Attrition"].values
y_sensitive = df["Gender"].values  # sensitive attribute: 0 = Female, 1 = Male

# Train-test split (stratify by the main target)
X_train, X_test, y_train, y_test, y_sensitive_train, y_sensitive_test = train_test_split(
    X, y_main, y_sensitive, test_size=0.2, random_state=42, stratify=y_main
)

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
y_sensitive_train_tensor = torch.tensor(y_sensitive_train, dtype=torch.long, device=device)
y_sensitive_test_tensor = torch.tensor(y_sensitive_test, dtype=torch.long, device=device)

# =============================
# 4. Define Hyperparameter Trials
# =============================
# Define 20 different trial configurations
trials = [
    {"lr": 0.001, "batch_size": 64, "lambda_adv": 0.5},
    {"lr": 0.001, "batch_size": 64, "lambda_adv": 1.0},
    {"lr": 0.001, "batch_size": 32, "lambda_adv": 0.5},
    {"lr": 0.001, "batch_size": 32, "lambda_adv": 1.0},
    {"lr": 0.0005, "batch_size": 64, "lambda_adv": 0.5},
    {"lr": 0.0005, "batch_size": 64, "lambda_adv": 1.0},
    {"lr": 0.0005, "batch_size": 32, "lambda_adv": 0.5},
    {"lr": 0.0005, "batch_size": 32, "lambda_adv": 1.0},
    {"lr": 0.0001, "batch_size": 64, "lambda_adv": 0.5},
    {"lr": 0.0001, "batch_size": 64, "lambda_adv": 1.0},
    {"lr": 0.0001, "batch_size": 32, "lambda_adv": 0.5},
    {"lr": 0.0001, "batch_size": 32, "lambda_adv": 1.0},
    {"lr": 0.005, "batch_size": 64, "lambda_adv": 0.5},
    {"lr": 0.005, "batch_size": 64, "lambda_adv": 1.0},
    {"lr": 0.005, "batch_size": 32, "lambda_adv": 0.5},
    {"lr": 0.005, "batch_size": 32, "lambda_adv": 1.0},
    {"lr": 0.01, "batch_size": 64, "lambda_adv": 0.5},
    {"lr": 0.01, "batch_size": 64, "lambda_adv": 1.0},
    {"lr": 0.01, "batch_size": 32, "lambda_adv": 0.5},
    {"lr": 0.01, "batch_size": 32, "lambda_adv": 1.0},
]

# =============================
# 5. Training & Evaluation Loop
# =============================
epochs = 50

best_trial = None
best_f1 = -1.0  # initialize best F1 score
results = []

for trial in trials:
    lr = trial["lr"]
    batch_size = trial["batch_size"]
    lambda_adv = trial["lambda_adv"]

    print(f"\n=== Starting trial: lr={lr}, batch_size={batch_size}, lambda_adv={lambda_adv} ===")

    # Initialize model for this trial
    input_size = X_train.shape[1]
    model = AdversarialFCNN(input_size, lambda_adv=lambda_adv).to(device)

    # Use class weighting for main task loss if desired
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion_main = nn.CrossEntropyLoss(weight=weights_tensor)
    criterion_adv = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_train = X_train_tensor.size(0)

    # Training loop for current trial
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train_tensor[indices]
            batch_y_main = y_train_tensor[indices]
            batch_y_sensitive = y_sensitive_train_tensor[indices]

            optimizer.zero_grad()
            logits_main, logits_adv = model(batch_x)
            loss_main = criterion_main(logits_main, batch_y_main)
            loss_adv = criterion_adv(logits_adv, batch_y_sensitive)

            loss = loss_main + loss_adv
            loss.backward()
            optimizer.step()

    # Evaluation on test set after training
    model.eval()
    with torch.no_grad():
        logits_main_test, _ = model(X_test_tensor)
        preds_main = torch.argmax(logits_main_test, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds_main)
    rec = recall_score(y_test, preds_main)
    f1 = f1_score(y_test, preds_main)

    trial_result = {
        "lr": lr,
        "batch_size": batch_size,
        "lambda_adv": lambda_adv,
        "accuracy": acc,
        "recall": rec,
        "f1": f1
    }
    results.append(trial_result)

    print(f"Trial Results -> Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_trial = trial_result

# =============================
# 6. Output Best Trial Results
# =============================
print("\n=== Best Trial ===")
print(f"Learning Rate: {best_trial['lr']}")
print(f"Batch Size: {best_trial['batch_size']}")
print(f"Lambda_adv: {best_trial['lambda_adv']}")
print(f"Accuracy: {best_trial['accuracy']:.4f}")
print(f"Recall: {best_trial['recall']:.4f}")
print(f"F1 Score: {best_trial['f1']:.4f}")

# Optionally, save all trial results to CSV for further analysis
results_df = pd.DataFrame(results)
results_df.to_csv("results/adversarial_hyperparameter_trials.csv", index=False)
print("✅ All trial results saved to 'results/adversarial_hyperparameter_trials.csv'")
