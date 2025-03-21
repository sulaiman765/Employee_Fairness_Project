import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score

import shap
import matplotlib
matplotlib.use('Agg')  # Non-blocking plot creation
import matplotlib.pyplot as plt

###########################################################
# 1. LOAD ADVANCED-PREPROCESSED DATA
###########################################################
TRAIN_DATA_PATH = "../results/train_amdn.csv"
df = pd.read_csv(TRAIN_DATA_PATH)

target_col = "Attrition"
if target_col not in df.columns:
    raise ValueError(f"'{target_col}' not found in CSV columns: {df.columns}")

X_full = df.drop(columns=[target_col]).values
y_full = df[target_col].values
feature_names = df.drop(columns=[target_col]).columns

print(f"✅ Loaded {len(df)} samples from '{TRAIN_DATA_PATH}'.")
print(f"Feature columns ({len(feature_names)}) : {list(feature_names)}")

###########################################################
# 2. TRAIN-VAL SPLIT
###########################################################
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    stratify=y_full,
    random_state=42
)
print(f"✅ Train split: {X_train_full.shape}, Validation split: {X_val_full.shape}")

# We apply standard scaling for the NN
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_val_full = scaler.transform(X_val_full)


###########################################################
# 3. FCNN MODEL DEFINITION
###########################################################
class AdvancedFCNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden1=128,
        hidden2=64,
        hidden3=32,
        dropout1=0.3,
        dropout2=0.2
    ):
        """
        A flexible FCNN with variable hidden sizes and dropout.
        """
        super(AdvancedFCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout1),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout2),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),

            nn.Linear(hidden3, 2)  # final: 2 logits
        )

    def forward(self, x):
        return self.net(x)

###########################################################
# 4. TRAINING FUNCTION (WITH EARLY STOPPING)
###########################################################
def train_and_evaluate_fcnn(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden1=128,
    hidden2=64,
    hidden3=32,
    dropout1=0.3,
    dropout2=0.2,
    lr=0.001,
    batch_size=32,
    epochs=100,
    patience=10,
    device="cpu"
):
    """
    Train + Evaluate an FCNN with early stopping.
    Returns: (model, (accuracy, recall, f1)) on validation
    """

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long, device=device)

    # Build model
    input_size = X_train.shape[1]
    model = AdvancedFCNN(
        input_size,
        hidden1=hidden1,
        hidden2=hidden2,
        hidden3=hidden3,
        dropout1=dropout1,
        dropout2=dropout2
    ).to(device)

    # Class weighting
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    def get_batches(X_data, y_data, batch_sz):
        n_samples = X_data.size(0)
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_sz):
            idx = perm[i:i+batch_sz]
            yield X_data[idx], y_data[idx]

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        # mini-batch
        for X_batch, y_batch in get_batches(X_train_t, y_train_t, batch_size):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        train_loss = total_train_loss / X_train_t.size(0)

        # validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # early stopping
                model.load_state_dict(best_model_state)
                break

    # Load best weights
    model.load_state_dict(best_model_state)
    model.eval()

    # Evaluate final
    with torch.no_grad():
        val_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()

    acc = accuracy_score(y_val, val_preds)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)

    return model, (acc, rec, f1)


###########################################################
# 5. DEFINE 20+ TRIALS FOR EXPERIMENTS
###########################################################
trials = [
    # (We'll vary LR, batch, dropout, hidden sizes, etc.)
    {"lr": 0.01,   "batch_size": 32, "dropout1": 0.3, "dropout2": 0.2, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.005,  "batch_size": 32, "dropout1": 0.3, "dropout2": 0.2, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.001,  "batch_size": 32, "dropout1": 0.3, "dropout2": 0.2, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.0005, "batch_size": 64, "dropout1": 0.3, "dropout2": 0.2, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.0005, "batch_size": 32, "dropout1": 0.4, "dropout2": 0.3, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.0001, "batch_size": 32, "dropout1": 0.2, "dropout2": 0.2, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.001,  "batch_size": 16, "dropout1": 0.3, "dropout2": 0.3, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.01,   "batch_size": 64, "dropout1": 0.2, "dropout2": 0.2, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.001,  "batch_size": 64, "dropout1": 0.3, "dropout2": 0.4, "hidden1":128, "hidden2":64,  "hidden3":32},
    {"lr": 0.01,   "batch_size": 32, "dropout1": 0.4, "dropout2": 0.4, "hidden1":128, "hidden2":64,  "hidden3":32},

    {"lr": 0.0001, "batch_size": 32, "dropout1": 0.4, "dropout2": 0.2, "hidden1":256, "hidden2":128, "hidden3":64},
    {"lr": 0.001,  "batch_size": 64, "dropout1": 0.2, "dropout2": 0.2, "hidden1":256, "hidden2":128, "hidden3":64},
    {"lr": 0.005,  "batch_size": 16, "dropout1": 0.3, "dropout2": 0.3, "hidden1":256, "hidden2":128, "hidden3":64},
    {"lr": 0.0005, "batch_size": 16, "dropout1": 0.4, "dropout2": 0.4, "hidden1":256, "hidden2":128, "hidden3":64},
    {"lr": 0.0001, "batch_size": 64, "dropout1": 0.3, "dropout2": 0.2, "hidden1":256, "hidden2":128, "hidden3":64},

    {"lr": 0.001,  "batch_size": 32, "dropout1": 0.2, "dropout2": 0.2, "hidden1":64,  "hidden2":64,  "hidden3":64},
    {"lr": 0.01,   "batch_size": 32, "dropout1": 0.2, "dropout2": 0.3, "hidden1":64,  "hidden2":64,  "hidden3":64},
    {"lr": 0.005,  "batch_size": 64, "dropout1": 0.3, "dropout2": 0.3, "hidden1":64,  "hidden2":64,  "hidden3":64},
    {"lr": 0.0005, "batch_size": 32, "dropout1": 0.2, "dropout2": 0.4, "hidden1":64,  "hidden2":64,  "hidden3":64},
    {"lr": 0.0001, "batch_size": 16, "dropout1": 0.4, "dropout2": 0.4, "hidden1":64,  "hidden2":64,  "hidden3":64},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model = None
best_score = (0.0, 0.0, 0.0)  # (acc, rec, f1)
best_params = None

for i, trial_params in enumerate(trials, start=1):
    print(f"\n=== Trial {i} / {len(trials)} ===")
    print("Hyperparams:", trial_params)

    model_current, (acc, rec, f1_val) = train_and_evaluate_fcnn(
        X_train_full,
        y_train_full,
        X_val_full,
        y_val_full,
        hidden1=trial_params["hidden1"],
        hidden2=trial_params["hidden2"],
        hidden3=trial_params["hidden3"],
        dropout1=trial_params["dropout1"],
        dropout2=trial_params["dropout2"],
        lr=trial_params["lr"],
        batch_size=trial_params["batch_size"],
        epochs=100,
        patience=10,
        device=device
    )

    print(f"Validation -> Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1_val:.4f}")

    # Track best by F1
    if f1_val > best_score[2]:
        best_score = (acc, rec, f1_val)
        best_model = model_current
        best_params = trial_params

print("\n=== Best Model Across All Trials ===")
print(f"Params: {best_params}")
print(f"Best Scores -> Accuracy: {best_score[0]:.4f}, Recall: {best_score[1]:.4f}, F1: {best_score[2]:.4f}")

##########################################################################
# 6. SHAP on the Best Model
##########################################################################
print("\n=== Computing SHAP for the Best Model ===")

try:
    def best_model_wrapper(x_np):
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = best_model(x_tensor).cpu().numpy()  # shape (n_samples, 2)
            return logits[:, 1]  # single logit for class=1

    # We'll do SHAP on the first 50 examples of the validation set
    X_shap = X_val_full[:50]
    background = shap.kmeans(X_train_full, 10)

    explainer = shap.KernelExplainer(best_model_wrapper, background)
    shap_values = explainer.shap_values(X_shap)  # shape (50, n_features)?

    print(f"shap_values.shape: {shap_values.shape}")
    print(f"X_shap.shape: {X_shap.shape}")

    plt.close("all")
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.savefig("shap_summary_plot_best_fcnn.png", dpi=300, bbox_inches="tight")
    print("✅ SHAP summary plot saved as 'shap_summary_plot_best_fcnn.png'")

except Exception as e:
    print(f"❌ SHAP for best FCNN failed: {e}")
    print("⚠️ Fallback: using RandomForestClassifier for SHAP.\n")

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_full, y_train_full)

    X_shap = X_val_full[:50]
    rf_explainer = shap.TreeExplainer(rf)
    shap_values_rf = rf_explainer.shap_values(X_shap)
    shap_values_class1_rf = shap_values_rf[1]

    plt.close("all")
    shap.summary_plot(
        shap_values_class1_rf,
        X_shap,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.savefig("shap_summary_plot_rf_fallback.png", dpi=300, bbox_inches="tight")
    print("✅ SHAP fallback plot saved as 'shap_summary_plot_rf_fallback.png'")

##########################################################################
# 7. SAVE BEST MODEL’S VAL PREDICTIONS FOR FAIRNESS ANALYSIS
##########################################################################
best_model.eval()
with torch.no_grad():
    val_outputs_best = best_model(
        torch.tensor(X_val_full, dtype=torch.float32, device=device)
    )
    val_preds_best = torch.argmax(val_outputs_best, dim=1).cpu().numpy()

val_preds_df = pd.DataFrame({"GroundTruth": y_val_full, "Prediction": val_preds_best})
os.makedirs("data", exist_ok=True)
val_preds_df.to_csv("data/val_predictions_best_fcnn.csv", index=False)

print("\n✅ Best model's validation predictions saved for fairness analysis in 'data/val_predictions_best_fcnn.csv'")
