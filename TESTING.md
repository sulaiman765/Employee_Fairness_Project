xvczvdxg# 🛠️ Testing & Debugging Log

## ✅ Week 1: Environment Setup & Data Preprocessing

### ❌ Issue: Virtual Environment Not Recognized  
**Problem:** After installing Python, `python --version` in PyCharm showed the wrong version.  
**Solution:** Fixed by selecting the correct interpreter (`.venv\Scripts\python.exe`) in PyCharm settings.  

### ❌ Issue: Dataset Not Loading in Pandas  
**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'`.  
**Solution:** Fixed by ensuring the dataset was placed in the correct `data/` folder and updating the file path.  

---

## ✅ Week 2: Model Training & Fairness Analysis

### ❌ Issue: Fairlearn ExponentiatedGradient Crashing  
**Problem:** `AssertionError: data can be loaded only once` when using `sensitive_features`.  
**Solution:** Fixed by ensuring `sensitive_features = X_train_df[["Gender"]]` **before calling `.fit()`**.  

### ❌ Issue: Logistic Regression Had Poor Recall on Attrition = 1  
**Problem:** Initial Logistic Regression model only had **13% recall**, meaning it missed most cases of employee attrition.  
**Solution:** Applied **SMOTE** to balance the dataset, and **ExponentiatedGradient** improved fairness, boosting recall to **61%**.  

---

## ✅ Week 3: Model Evaluation & Git Issues

### ❌ Issue: GitHub Not Tracking `Python Scripts/` Folder  
**Problem:** Git was ignoring `Python Scripts/` despite multiple commits.  
**Solution:** Fixed by renaming `Python Scripts/` to `src/`, pushing the changes, and renaming it back.  

### ❌ Issue: Git Push Showing "Everything up-to-date" But No Files on GitHub  
**Problem:** Git claimed everything was pushed, but `Python Scripts/` was missing on GitHub.  
**Solution:** Ran `git rm -r --cached "Python Scripts/"`, then re-added and committed the files.  

---

## ✅ Week 4: Fairness Mitigation & Model Adjustments  

### ❌ Issue: Random Forest Had Poor Fairness Performance  
**Problem:** **Fairlearn evaluation showed significant bias**, especially in recall for different demographic groups.  
**Solution:** Used **ExponentiatedGradient with Equalized Odds**, which improved fairness scores.  

### ❌ Issue: Post-Processing Not Explicitly Implemented  
**Problem:** We didn’t apply post-processing like `ThresholdOptimizer`.  
**Solution:** Realized that **ExponentiatedGradient already adjusts predictions**, so additional post-processing was unnecessary.  

---

✅ **We will continue updating this file as we face and solve new issues in later phases!**
# 🛠️ Testing & Debugging Log

## ✅ Week 1: Initial Setup & No Recorded Model Results  
During Week 1, we focused on:  
- Selecting the dataset (`processed_employee_attrition.csv`).  
- Installing essential libraries (Fairlearn, SHAP, DoWhy, PyTorch).  
- Performing initial EDA (class imbalance, feature distributions).  

🚀 **No model results were recorded during this phase. The first evaluation was in Week 2.**  

---

# 📈 Results Improvement Log  

## ✅ Week 2: Initial Model Performance  

### 🔹 Logistic Regression (Before Fairness Improvements)  
**Accuracy:** 79.59%  
**Recall (Attrition = 1):** 13%  

### 🔹 Random Forest (Before Fairness Improvements)  
**Accuracy:** 80.61%  
**Recall (Attrition = 1):** 25.53%  

**Key Problem:**  
❌ The model failed to correctly classify employee attrition cases (very low recall).  

---

## ✅ Week 3 & 4: Applying SMOTE, Fairness Constraints, and Model Adjustments  

### 🔹 Logistic Regression (After SMOTE & ExponentiatedGradient)  
**Accuracy:** 78.91%  
**Recall (Attrition = 1):** 61.70%  

### 🔹 Random Forest (After SMOTE & ExponentiatedGradient)  
**Accuracy:** 80.61%  
**Recall (Attrition = 1):** 25.53% (No improvement)  

**Key Fixes & Code Changes:**  
🚀 **We applied SMOTE to balance the dataset, which improved recall.**  
🚀 **We added `ExponentiatedGradient` to train the model with fairness constraints.**  
🚀 **We fixed an issue with `sensitive_features` formatting that was causing errors.**  

🔹 **SMOTE (Before & After)**  
```python
# ❌ Before (Without SMOTE, Poor Recall)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ After (With SMOTE, Better Recall for Attrition Cases)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ❌ Before (No Fairness Constraints, Unfair Model)
logistic_model.fit(X_train, y_train)

# ✅ After (Fairness Constraints Applied, More Fair Predictions)
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
logistic_fair = ExponentiatedGradient(logistic_model, constraints=EqualizedOdds())
logistic_fair.fit(X_train, y_train, sensitive_features=X_train_df[["Gender"]])

# ❌ Before (Error in Fairlearn Constraints, Caused `AssertionError`)
logistic_fair.fit(X_train, y_train, sensitive_features=X_train["Gender"])

# ✅ After (Fixed by Converting `sensitive_features` to a DataFrame)
sensitive_features = X_train_df[["Gender"]]  # Fix: Ensuring correct format
logistic_fair.fit(X_train, y_train, sensitive_features=sensitive_features)

## ✅ Week 5: Deep Learning (FCNN) vs. Traditional Models  

### 🔹 FCNN (Before Fixes)  
**Accuracy:** 84.01%  
**Recall (Attrition = 1):** 0.00% ❌  

### 🔹 FCNN (After Applying SMOTE & Weighted Loss)  
**Accuracy:** 77.21%  
**Recall (Attrition = 1):** 70.21% ✅  

### **Key Fixes & Code Changes**  
🚀 **We applied SMOTE to balance the dataset, which improved recall.**  
🚀 **We used a weighted loss function so the model pays more attention to minority class samples (Attrition = 1).**  
🚀 **This significantly improved recall while keeping accuracy stable.**  

### **🔹 SMOTE (Before & After)**  
```python
# ❌ Before (Without SMOTE, Poor Recall)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ After (With SMOTE, Better Recall for Attrition Cases)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ❌ Before (Equal loss for both classes)
criterion = nn.CrossEntropyLoss()

# ✅ After (Weighted loss to focus on Attrition cases)
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=weights)

# **Further Added Improvements for Week 5**

## ✅ 1. SMOTE - Handling Class Imbalance  
```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

What it does: Balances the dataset by oversampling the minority class (employees who left).
Why it improves fairness: Ensures that the model doesn't favor the majority class (employees who stayed), preventing biased predictions.



class_counts = np.bincount(y_train)  # Count instances of each class
class_weights = 1.0 / class_counts   # Compute inverse class frequency
weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

# Use weighted loss function
criterion = nn.CrossEntropyLoss(weight=weights)

What it does: Penalizes misclassification of underrepresented classes more heavily.
Why it improves fairness: Helps prevent the model from ignoring minority classes and encourages balanced predictions.

import shap

# Define model wrapper for SHAP
def model_wrapper(x):
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()

# Reduce SHAP background dataset using k-means
background = shap.kmeans(X_train, 10)

# SHAP KernelExplainer
explainer = shap.KernelExplainer(model_wrapper, background)
shap_values = explainer.shap_values(X_test[:50])  # Compute SHAP values for 50 test samples

What it does: Uses SHAP to analyze feature importance and detect bias in predictions.
Why it improves fairness: Helps identify which features contribute to unfair model predictions, enabling bias mitigation strategies.

class FCNN(nn.Module):
    def __init__(self, input_size):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)  # Output layer (logits, no Softmax)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)  # Return raw logits
        return x

What it does: Removes Softmax activation so SHAP can correctly interpret raw model outputs.
Why it improves fairness: Prevents SHAP from misrepresenting feature importance, leading to more accurate bias detection.

from sklearn.ensemble import RandomForestClassifier

try:
    # If SHAP fails, use a fallback model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test[:50])

    shap_values_fixed = np.array(shap_values[1])  # Use class 1 SHAP values

except Exception as e:
    print(f"❌ SHAP computation failed: {str(e)}")

What it does: Uses a fallback RandomForest model for SHAP when deep learning explainability fails.
Why it improves fairness: Ensures that feature importance can still be analyzed even if SHAP struggles with the FCNN.

y_test_df = pd.DataFrame({"Attrition": y_test})
y_pred_df = pd.DataFrame({"Predicted": predictions})

import os
os.makedirs("data", exist_ok=True)

y_test_df.to_csv("data/y_test.csv", index=False)
y_pred_df.to_csv("data/y_pred.csv", index=False)

What it does: Saves model predictions for later fairness analysis in fairness_evaluation.py.
Why it improves fairness: Enables measurement of fairness metrics like Demographic Parity Difference and Equalized Odds in Week 6.