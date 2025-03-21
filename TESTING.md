# 🛠️ Testing & Debugging Log

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

# Detailed Fairness Improvement for Baseline vs Improved FCNN Model:

We started with a Baseline FCNN (which had no fairness optimizations) and improved it with six key fairness techniques to reduce bias.
This report shows the exact code changes, how they impact bias reduction, and compares the results.

2️⃣ What Was Changed in the Code? (With Side-by-Side Comparisons & Bias Effects)

✅ 1. SMOTE - Handling Class Imbalance

🔴 Baseline FCNN (No SMOTE, Imbalanced Data)
# NO SMOTE APPLIED
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

🟢 Improved FCNN (SMOTE Applied to Balance Classes)
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

How This Change Reduces Bias
Problem in Baseline:

The dataset was highly imbalanced (fewer attrition cases).
The model ignored attrition cases because predicting "stay" minimized loss.
Effect of Change:

SMOTE creates synthetic minority-class samples, forcing the model to learn patterns related to attrition.
This ensures equal learning for both classes, preventing class imbalance bias.
✅ Bias Reduction Impact:
✔️ Increases detection of attrition cases.
✔️ Prevents the model from favoring employees who stayed.

✅ 2. Class Weighting in Loss Function - Adjusting Model Training

🔴 Baseline FCNN (Equal Loss for Both Classes)
criterion = nn.CrossEntropyLoss()

🟢 Improved FCNN (Class-Weighted Loss Function)
# Compute class weights
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts  
weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

# Use weighted loss function
criterion = nn.CrossEntropyLoss(weight=weights)

How This Change Reduces Bias
Problem in Baseline:

Loss function treated misclassifications equally, meaning the model ignored rare attrition cases.
Effect of Change:

Increases the penalty for misclassifying attrition cases, forcing the model to pay more attention to them.
Ensures the model doesn’t just optimize for accuracy but also fairness.
✅ Bias Reduction Impact:
✔️ Prevents the model from ignoring underrepresented classes.
✔️ Improves predictions for employees likely to leave.

✅ 3. Removing Softmax from FCNN - Ensuring SHAP Accuracy

🔴 Baseline FCNN (Softmax Applied)
self.fc3 = nn.Linear(32, 2)
self.softmax = nn.Softmax(dim=1)  # Softmax applied

def forward(self, x):
    x = self.relu1(self.fc1(x))
    x = self.relu2(self.fc2(x))
    x = self.softmax(self.fc3(x))  # Returns probabilities
    return x

🟢 Improved FCNN (Softmax Removed for SHAP Accuracy)
self.fc3 = nn.Linear(32, 2)  # No Softmax

def forward(self, x):
    x = self.relu1(self.fc1(x))
    x = self.relu2(self.fc2(x))
    x = self.fc3(x)  # Returns raw logits
    return x

How This Change Reduces Bias
Problem in Baseline:

SHAP (used to explain predictions) was misinterpreting feature importance due to Softmax.
Effect of Change:

Ensures SHAP correctly interprets the model’s raw outputs, allowing accurate bias detection.
✅ Bias Reduction Impact:
✔️ Improves interpretability of model predictions.
✔️ Ensures feature importance is correctly evaluated, helping in further bias mitigation.

✅ 4. SHAP Integration for Bias Detection

🔴 Baseline FCNN (No SHAP Analysis)
# No SHAP analysis in baseline model

🟢 Improved FCNN (SHAP Added for Bias Detection)
import shap

# Define model wrapper
def model_wrapper(x):
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()

# Reduce SHAP background dataset using k-means
background = shap.kmeans(X_train, 10)

# Compute SHAP values
explainer = shap.KernelExplainer(model_wrapper, background)
shap_values = explainer.shap_values(X_test[:50]) 

How This Change Reduces Bias
Problem in Baseline:

No way to understand which features were unfairly influencing predictions.
Effect of Change:

SHAP identifies if gender, overtime, or business travel unfairly affect decisions.
✅ Bias Reduction Impact:
✔️ Ensures fairness interventions actually work.
✔️ Helps detect hidden bias in the model’s decision process.

✅ 5. Fallback RandomForest Model for SHAP - Improving Explainability

🔴 Baseline FCNN (No Fallback for SHAP)
# No backup model if SHAP fails

🟢 Improved FCNN (RandomForest Backup for SHAP)
from sklearn.ensemble import RandomForestClassifier

# Fallback model in case SHAP fails
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test[:50])

🔍 How This Change Reduces Bias
Problem in Baseline:

If SHAP failed, there was no way to check model fairness.
Effect of Change:

Ensures feature analysis is always available, making bias detection reliable.
✅ Bias Reduction Impact:
✔️ Guarantees interpretability even if FCNN is complex.
✔️ Ensures fairness evaluation is always possible.

✅ 6. Saving Predictions for Fairness Analysis

🔴 Baseline FCNN (No Predictions Saved)
# No baseline fairness comparison

🟢 Improved FCNN (Predictions Saved for Fairness Evaluation)
y_test_df = pd.DataFrame({"Attrition": y_test})
y_pred_df = pd.DataFrame({"Predicted": predictions})

import os
os.makedirs("data", exist_ok=True)

y_test_df.to_csv("data/y_test_fcnn_improved.csv", index=False)
y_pred_df.to_csv("data/y_pred_fcnn_improved.csv", index=False)

How This Change Reduces Bias
Problem in Baseline:

No stored predictions to compare bias before and after improvements.
Effect of Change:

Now, we can compute fairness metrics and compare models directly.
✅ Bias Reduction Impact:
✔️ Enables direct fairness comparison between baseline and improved models.
✔️ Provides clear evidence of fairness improvements.

Key Takeaways:
✅ Baseline FCNN had recall = 0.0, meaning it failed to predict attrition cases.
✅ Improved FCNN increased recall, meaning it now identifies attrition cases.
✅ Bias is significantly reduced while maintaining strong accuracy.

🚀 Final Verdict: The improved FCNN is significantly fairer and more effective! 🚀

Multiple Hyperparameter Trials: Detailed Explanation & Results
Overview
We conducted 20 distinct training trials for our FCNN, each with different hyperparameters (learning rate, batch size, dropout, and hidden-layer sizes). The primary goal is to find a configuration that maximizes key metrics:

Accuracy – The percentage of correct predictions overall.
Recall – The proportion of “Yes” (minority class) cases correctly identified (critical for imbalanced data).
F1 Score – The harmonic mean of precision and recall, balancing both.
By systematically altering training parameters across 20 trials, we increase the likelihood of uncovering a combination that yields optimal performance. Below, we break down each hyperparameter and how it influences our final metrics.

1. Hyperparameters Explored
1.1 Learning Rate (lr)
Definition: Controls how fast or slow the model’s weights are updated during backpropagation.
Values used across the 20 trials may include 0.01, 0.005, 0.001, 0.0005, 0.0001.
Impact:
High LR (e.g., 0.01): Faster convergence initially but can skip over optimal minima or overshoot. Potentially improves accuracy early but might degrade stability.
Low LR (e.g., 0.0001): Slower, more careful weight updates – can achieve a more stable convergence. Often beneficial for recall if the minority class patterns are subtle.
1.2 Batch Size (batch_size)
Definition: Number of samples processed before the model updates its weights.
Values might include 16, 32, 64.
Impact:
Smaller Batches (16, 32): Introduce more noise into gradients, which can help escape local minima and often improves recall, since the model sees “Yes” examples more frequently in each epoch.
Larger Batches (64+): More stable gradient estimates each step but risk overlooking minority patterns if your dataset is highly imbalanced.
1.3 Dropout (dropout1, dropout2)
Definition: Randomly drops units (neurons) during training to reduce overfitting.
Values commonly range from 0.2 to 0.4.
Impact:
Higher Dropout (0.3–0.4): Helps prevent overfitting by forcing the network to generalize. Might slow down convergence but can stabilize recall and accuracy.
Lower Dropout (0.2 or less): Retains capacity, may learn patterns faster—but if data is not sufficiently large or is noisy, can lead to overfitting, harming recall on the minority class.
1.4 Hidden Layer Sizes (hidden1, hidden2, hidden3)
Definition: Number of neurons in each layer of the FCNN, e.g., (128 → 64 → 32) or (256 → 128 → 64) or (64 → 64 → 64).
Impact:
Larger Layers (e.g., 256 → 128 → 64): Greater capacity to learn complex relationships, especially beneficial for capturing minority-class nuances, boosting recall and possibly F1.
Smaller Layers (e.g., 64 → 64 → 64): Less capacity but typically faster training and less overfitting risk – can be helpful if the dataset is not too large or the distribution is straightforward.
2. How These Trials Were Structured
Systematic Variation: Each trial picks a unique combination of (lr, batch_size, dropout1, dropout2, hidden1, hidden2, hidden3).
Train with Early Stopping: Each model trains for up to 100 epochs but can stop earlier if the validation loss fails to improve, preventing overfitting.
Evaluate on Validation Set: After each trial, we compute accuracy, recall, and F1 on a hold-out validation set.
Compare & Select: We keep track of the highest F1 (or whichever metric is your priority). The top performer is the “best” configuration.
3. Impact on Accuracy, Recall, and F1
3.1 Why Accuracy Improves
Appropriate LR + dropout can balance between underfitting and overfitting, yielding a stable decision boundary that correctly classifies most examples.
Batch size changes can improve overall generalization, leading to more correct predictions on both majority and minority classes.
3.2 Why Recall Improves
Minority Class Representation: With carefully chosen hyperparameters (e.g., moderate LR, small batch size, enough capacity), the network sees and learns minority patterns effectively.
Regularization: Dropout prevents memorizing only the majority patterns, so the model better captures important minority signals.
3.3 Why F1 Improves
Balancing Precision & Recall: The best hyperparam combination ensures we don’t trade off one for the other.
If the model overfits the majority class, recall suffers. If it underfits, accuracy suffers. Correct hyperparams let the model handle both classes well, thus raising F1.
4. Final Results & Takeaways
Reduced Bias: Varying hyperparams helps the network avoid ignoring the minority class; specifically, smaller batch sizes and moderate dropout often yield higher recall.
Improved Generalization: The best trials found an optimal synergy of learning rate and dropout, which balanced learning speed with robust generalization.
Iterative Process: Trying 20 trials isn’t the end; it’s a systematic sampling of the hyperparameter space. If needed, you can further tune around the best performers.
In conclusion, running these 20 hyperparameter trials was crucial for achieving better accuracy, higher recall, and stronger F1. By systematically adjusting learning rate, batch size, dropout, and hidden sizes, we found a configuration that maximizes performance on both the majority and minority classes, thereby reducing bias and improving overall model quality.

1. Project Structure Overview
Let’s start by summarizing the purpose of each file and how they interact:

Core Files:
eda.py:

Performs exploratory data analysis (EDA) on the dataset.

Visualizes class distributions, correlations, and attrition by gender/overtime.

Output: Insights into dataset structure and potential biases.

preprocessing.py:

Cleans and preprocesses the raw dataset.

Handles categorical encoding, normalization, and feature scaling.

Output: processed_employee_attrition.csv.

new_preprocessing.py:

Implements advanced preprocessing techniques like Adaptive Multi-Group Distribution Normalization (AMDN).

Balances the dataset using SMOTE and handles sensitive attributes.

Output: train_amdn.csv.

train_models.py:

Trains baseline models (Logistic Regression, Random Forest).

Applies SMOTE for class imbalance and evaluates model performance.

Output: Trained models and predictions (y_test.csv, y_pred.csv).

train_fcnn_baseline.py:

Trains a baseline Fully Connected Neural Network (FCNN) without fairness improvements.

Output: Baseline FCNN model and predictions.

train_fcnn_improved.py:

Trains an improved FCNN with fairness techniques (SMOTE, class weighting, SHAP).

Output: Improved FCNN model and predictions.

train_fcnn_with_shap_for_new_preprocessing.py:

Trains an FCNN using the advanced preprocessed dataset (train_amdn.csv).

Includes SHAP explainability and fallback RandomForest for SHAP.

Output: Best FCNN model, SHAP plots, and predictions.

fairness_analysis.py:

Computes fairness metrics (Demographic Parity, Equal Opportunity) for model predictions.

Output: Fairness metrics for baseline and improved models.

fairness_evaluation.py:

Compares fairness metrics between baseline and improved models.

Output: Fairness comparison results (fairness_comparison.csv).

fairness_mitigation.py:

Implements fairness mitigation techniques (ExponentiatedGradient, Equalized Odds).

Output: Fairness-aware models and predictions.

permutation_importance.py:

Computes feature importance using permutation importance.

Output: Feature importance rankings and visualizations.

TESTING.md:

Documents testing, debugging, and fairness improvements.

Tracks issues, solutions, and model performance over time.

What is utils.py?
utils.py is a shared utility file that contains reusable functions and classes used across multiple scripts in your project. Instead of writing the same code repeatedly in different files, you centralize it in utils.py and import it wherever needed.

What Will utils.py Do?
Here’s what utils.py will include:

Data Loading:

A function to load and preprocess the dataset (e.g., load_data).

Data Preprocessing:

A function to split the data, apply SMOTE, and normalize features (e.g., preprocess_data).

Model Definition:

A class for the FCNN model (e.g., FCNN).

Common Utilities:

Functions for saving/loading models, computing metrics, etc.

Why Refactor Code into utils.py?
1. Avoid Code Duplication
Instead of writing the same code in train_fcnn_baseline.py, train_fcnn_improved.py, and other files, you write it once in utils.py and reuse it.

Example: Both train_fcnn_baseline.py and train_fcnn_improved.py load data, preprocess it, and define the FCNN model. This code can be moved to utils.py.

2. Improve Readability
By moving reusable code to utils.py, your main scripts (e.g., train_fcnn_baseline.py) become shorter and easier to read.

Example: Instead of 100 lines of data loading and preprocessing, you’ll have just one line: X_train, X_test, y_train, y_test = preprocess_data(X, y).

3. Easier Maintenance
If you need to update the data loading or preprocessing logic, you only need to change it in one place (utils.py) instead of updating multiple files.

Example: If you decide to change the normalization method, you only update preprocess_data in utils.py.

4. Promote Modularity
Each script focuses on a specific task (e.g., training, evaluation) and relies on utils.py for common functionality.

Example: train_fcnn_baseline.py focuses on training the baseline model, while utils.py handles data loading and preprocessing.

Further Analysis of Model Performance and Fairness
In this project, we evaluated three model types using both standard performance and fairness metrics. Below is a summary of our findings and conclusions:

Model Comparison
1. Baseline FCNN
Overall Accuracy: ~84.0%
Recall (Minority Class - Attrition = 1): 0.0%
Observations:
Although the baseline FCNN achieves high overall accuracy, it completely fails to predict any positive cases. This indicates that the model is heavily biased toward the majority class, rendering it ineffective for identifying attrition—an especially critical concern given our fairness goals.
2. RandomForest Model (from train_models.py)
Overall Accuracy: ~80.6%
Recall (Minority Class): ~25.5%
Group Fairness Analysis:
Subgroup performance varies notably. For example, one group (Gender=1, OverTime=1) has lower accuracy (~59.1%) and recall (~19.0%), while another (Gender=1, OverTime=0) achieves higher accuracy (~88.1%) and recall (~30.0%).
Observations:
The RandomForest model shows some capability to predict the minority class compared to the baseline FCNN, but the overall recall remains low. Additionally, there are significant disparities across sensitive subgroups, indicating uneven model behavior.
3. Improved FCNN
Overall Accuracy: ~77.2%
Recall (Minority Class): ~65.96%
Observations:
Despite a slight drop in overall accuracy, the improved FCNN model substantially increases recall for attrition cases. This suggests a better balance in identifying positive cases, which is critical when reducing bias is a priority. The model's enhanced recall comes at the expense of a modest decrease in accuracy—a trade-off that is often acceptable when addressing fairness in imbalanced datasets.
Overall Ranking
When considering both standard performance and fairness, the models are ranked as follows:

Improved FCNN:
Pros: Significantly higher recall for the minority class (65.96%) indicates effective identification of attrition cases.
Cons: Slightly lower overall accuracy (~77.2%), but this is acceptable given the improved fairness.
RandomForest Model:
Pros: Moderate overall accuracy (80.6%) with nonzero recall (25.5%).
Cons: Notable disparities in performance across sensitive groups, and the recall for attrition remains relatively low.
Baseline FCNN:
Pros: High overall accuracy (~84.0%).
Cons: Zero recall for attrition makes it unsuitable despite the high accuracy.
Conclusions
Fairness vs. Accuracy Trade-off:
The baseline FCNN, while accurate overall, is inadequate because it fails to capture the minority class. In contrast, the improved FCNN sacrifices some overall accuracy to achieve a much higher recall, thereby reducing bias and ensuring more equitable outcomes across sensitive groups.

Importance of Group-Level Analysis:
Fairness metrics by subgroup reveal disparities that are not evident when only looking at aggregate performance. For instance, the RandomForest model shows varying performance among groups defined by Gender and OverTime, highlighting the need for targeted fairness interventions.

Recommendation:
For applications where correctly identifying attrition is critical (and where fairness is a primary concern), the Improved FCNN is the most suitable model despite its lower overall accuracy. It demonstrates a more balanced performance and a better trade-off between fairness and predictive power.

