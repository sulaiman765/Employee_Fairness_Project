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

Detailed Fairness Analysis Report
Our fairness analysis is structured into two parts: the existing fairness features and the new additional fairness metrics and visualizations. This dual approach provides both a high-level view and a more nuanced insight into the fairness performance of our models.

1. Existing Fairness Features
Using Fairlearn’s MetricFrame, we compute several standard performance metrics for the overall dataset and for specific subgroups defined by sensitive attributes (in our case, Gender and OverTime). These metrics include:

Accuracy:
The proportion of correct predictions (both positives and negatives) over the total number of predictions. It gives a general measure of model performance.

Precision:
The ratio of true positive predictions to the total predicted positives. This metric tells us how reliable a positive prediction is, but in our case, it is particularly useful when combined with recall.

Recall (True Positive Rate):
The ratio of correctly predicted positive observations to all actual positives. Recall is especially critical in our setting, as it shows how effectively the model identifies attrition cases (the minority class).

F1 Score:
The harmonic mean of precision and recall. F1 score provides a balanced measure that takes both false positives and false negatives into account.

Balanced Accuracy:
The average of recall obtained on each class. This metric is particularly valuable for imbalanced datasets since it accounts for unequal class distribution.

Using these metrics, we can evaluate our model both in aggregate and for each subgroup. For example, in our best adversarial model, we observed overall values such as:

Accuracy ≈ 86.05%
Recall ≈ 51.06%
F1 Score ≈ 53.93%
When we break these metrics down by subgroup (e.g., by Gender and OverTime combinations), we may notice differences. For instance:

For Gender=0, OverTime=0, the accuracy might be high but the recall might be lower.
Conversely, for Gender=1, OverTime=1, the model might exhibit much higher recall and F1 score.
These group-level metrics help us identify disparities—indicating that the model's performance is not uniform across all demographic groups.

2. New Additional Fairness Metrics and Visualizations
To supplement the standard metrics, we introduced additional fairness metrics that focus on quantifying disparities in the model’s predictions. These include:

Demographic Parity Difference:

Definition: The difference in the positive prediction rate between two groups.
Interpretation: A value close to 0 indicates that both groups receive similar rates of positive predictions, suggesting equitable treatment.
Our Result: Approximately 0.0509, suggesting a small difference in positive prediction rates.
Equal Opportunity Difference:

Definition: The difference in true positive rates (recall) between groups.
Interpretation: This metric focuses on the fairness of correctly identifying the minority class for each group. A larger difference indicates that one group is at a disadvantage in terms of having their positive outcomes recognized.
Our Result: Approximately 0.1109, indicating that one group has a modest advantage over the other in correctly identifying positives.
Average Odds Difference:

Definition: The average of the differences in true positive rates and false positive rates between groups.
Interpretation: It provides a balanced view of performance disparities, taking into account both types of errors. A lower value means the error rates are similar across groups.
Our Result: Approximately 0.0663, which suggests a moderate disparity.
Disparate Impact Ratio:

Definition: The ratio of the positive prediction rate for the unprivileged group to that of the privileged group.
Interpretation: A value near 1 indicates parity between groups, while values far from 1 suggest potential bias.
Our Result: Approximately 1.4538, implying that one group is about 45% more likely to receive a positive prediction than the other—a disparity that may warrant further investigation.
Visualizations
To complement these numerical metrics, we generated visualizations that display these additional fairness metrics for the sensitive feature "Gender." The visualizations include:

Bar Chart for Demographic Parity:
This chart shows the positive prediction rate for each gender group, helping visualize whether the rates are similar.

Bar Chart for Equal Opportunity (TPR):
This chart displays the true positive rate for each group, illustrating any differences in the model's ability to correctly identify positive cases.

Bar Chart for False Positive Rate (FPR):
This chart highlights the false positive rate for each group, which is important for understanding if one group is more likely to be incorrectly labeled as positive.

These visualizations provide an immediate, intuitive view of the disparities between groups, supporting our quantitative findings and making it easier to communicate the fairness performance of our model.

3. Summary
Existing Features:
Our initial fairness analysis using Fairlearn’s MetricFrame provided group-level accuracy, precision, recall, F1 score, and balanced accuracy. This helped us understand overall performance and detect disparities between subgroups.

New Additional Metrics:
We extended the fairness analysis by introducing additional metrics (Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, and Disparate Impact Ratio) that more directly quantify the differences in outcomes between sensitive groups.

Visualizations:
The bar charts for positive prediction rates, true positive rates, and false positive rates allow us to visually inspect these disparities, providing a clearer picture of where our model may be exhibiting bias.

Conclusions:
While our best adversarial model achieves a high overall accuracy (86.05%) and a reasonable recall (51.06%), the additional metrics and visualizations reveal that disparities still exist among subgroups. For example, one subgroup (e.g., Gender=1, OverTime=0) has notably lower performance compared to another (e.g., Gender=1, OverTime=1). These insights are critical for guiding future fairness refinements and ensuring that our model's decisions are equitable across all groups.

This comprehensive review of our fairness analysis will be included in TESTING.md as part of our documentation and report, ensuring that both the existing and additional fairness measures are well-documented and understood.

Visualizing Additional Fairness Metrics
To gain deeper insight into how our adversarial model treats different gender groups, we plotted three additional fairness metrics for the Gender feature:

Demographic Parity per Group (Positive Prediction Rate)
Equal Opportunity per Group (True Positive Rate)
False Positive Rate per Group

Sample 3-panel bar chart showing Positive Prediction Rate (Demographic Parity), True Positive Rate (Equal Opportunity), and False Positive Rate for each gender group.

Demographic Parity per Group (Left Chart)

Definition: The proportion of samples predicted as positive (i.e., receiving the “1” label) in each group.
Observation: The bar for Gender=0 is slightly lower than for Gender=1. This implies that one group is receiving more positive predictions than the other.
Interpretation: A difference here suggests that, overall, one gender might be favored or disfavored in terms of positive classifications.
Equal Opportunity per Group (Middle Chart)

Definition: The true positive rate (TPR) for each group, which is also the recall for positive cases in that group.
Observation: Gender=1 (right bar) might have a slightly higher TPR than Gender=0 (left bar).
Interpretation: A higher TPR means that the model is more likely to correctly identify positive outcomes for that group. If the difference is large, it indicates potential unfairness in how positives are detected between groups.
False Positive Rate per Group (Right Chart)

Definition: The fraction of negative cases that are incorrectly classified as positive within each group.
Observation: If Gender=1 shows a higher FPR than Gender=0 (or vice versa), it means that group is more often incorrectly labeled as positive.
Interpretation: A difference here indicates the model is making more false alarms for one group compared to the other.
Key Takeaways
Demographic Parity Difference:
The difference in positive prediction rates between groups is a direct measure of whether one group is receiving systematically more positive classifications.

Equal Opportunity Difference:
By focusing on TPR (recall) for each group, we see how well the model identifies actual positives in each demographic. A substantial difference signals unequal treatment in identifying positives.

False Positive Rate Disparities:
Groups with higher FPR are more often falsely flagged as positive. This can have significant implications, depending on the context of the model’s usage.

Overall, these bar charts reinforce the numerical findings of the Demographic Parity Difference, Equal Opportunity Difference, and Average Odds Difference computed in the code. They provide a quick visual confirmation of any disparities and make it easier to communicate these findings to stakeholders who may not be as comfortable parsing raw metrics.

Final Results: Advanced FCNN with SHAP & AMDN Preprocessing
Best Hyperparameter Trial
Learning Rate: 0.0001
Batch Size: 32
Lambda_adv: 0.5 (Adversarial component)
Accuracy: 82.31%
Recall: 59.57%
F1 Score: 51.85%
These results indicate that the best model, under the given hyperparameter configurations, achieved a moderate trade‐off between accuracy and recall. The slightly lower F1 score suggests there is still room for improvement in balancing precision and recall.

Standard Performance & Fairness Metrics
Standard Metrics (Overall):

Accuracy: ~82.31%
Precision: ~45.90%
Recall: ~59.57%
F1 Score: ~51.85%
Fairlearn Group-Level Metrics:
By breaking down performance by (Gender, OverTime) groups, we observe:

(Gender=0, OverTime=0):

Accuracy: 91.67%
Precision: 50.00%
Recall: 42.86%
F1 Score: 46.15%
(Gender=0, OverTime=1):

Accuracy: 71.88%
Precision: 50.00%
Recall: 66.67%
F1 Score: 57.14%
(Gender=1, OverTime=0):

Accuracy: 80.60%
Precision: 10.00%
Recall: 20.00%
F1 Score: 13.33%
(Gender=1, OverTime=1):

Accuracy: 77.27%
Precision: 73.91%
Recall: 80.95%
F1 Score: 77.27%
These subgroup metrics show significant variability. In particular, the subgroup (Gender=1, OverTime=0) has low precision and recall, indicating the model struggles more with that demographic slice.

Additional Fairness Metrics
Demographic Parity Difference: ~0.0864

Interpretation: One gender group receives positive predictions at a rate ~8.64% higher than the other.
Equal Opportunity Difference: ~0.0504

Interpretation: The difference in recall (true positive rate) between the two gender groups is ~5.04%.
Average Odds Difference: ~0.0618

Interpretation: This averages the differences in both true positive rate and false positive rate across groups, indicating a moderate level of disparity (~6.18%).
Disparate Impact Ratio: ~1.5568

Interpretation: One group is about 55.68% more likely to receive a positive prediction than the other, pointing to potential bias.
Visualizations

Example 3-panel chart showing Demographic Parity (Positive Prediction Rate), Equal Opportunity (True Positive Rate), and False Positive Rate for each gender group.

Left (Demographic Parity):
Shows the positive prediction rate is higher for Gender=1 than for Gender=0, consistent with the Demographic Parity Difference above.
Center (Equal Opportunity):
True positive rate also favors Gender=1, though not by as large a margin.
Right (False Positive Rate):
The false positive rates vary, further highlighting differences in error distribution.
Key Takeaways
Improved Recall:
Compared to baseline FCNNs, recall is higher (59.57%), indicating that more positive (attrition) cases are being caught.
Subgroup Disparities:
Significant differences persist for certain subgroups, especially (Gender=1, OverTime=0).
Moderate Bias Indicators:
Additional metrics (Demographic Parity Difference, Equal Opportunity Difference, etc.) confirm there is still bias to address, but the model is more balanced than some earlier baselines.
Overall, this advanced FCNN with SHAP explainability and AMDN preprocessing is a notable step forward, providing a decent balance of performance and fairness. However, the subgroup metrics reveal ongoing challenges in equitable treatment, pointing to possible avenues for further fairness mitigation or hyperparameter refinement.

Review of FCNN Adversarial Learning Model
Overview
The FCNN Adversarial Learning model was developed to mitigate bias in employee attrition predictions by reducing the influence of the sensitive attribute Gender on the model’s decisions. The model integrates an adversarial branch into the main network using a gradient reversal layer. The adversary branch is tasked with predicting Gender, while the gradient reversal mechanism forces the shared feature representation to be less predictive of Gender. Additionally, advanced preprocessing using Adaptive Multi-Group Distribution Normalization (AMDN) was applied to further align feature distributions across groups. A comprehensive hyperparameter tuning loop was used to optimize learning rate, batch size, and the adversarial weighting parameter (λ), with the best trial selected based on the F1 score.

Key Components
Gradient Reversal Layer:
In this model, a gradient reversal layer is inserted before the adversary branch. During the forward pass, it passes the data unchanged. However, during backpropagation, it multiplies the gradients by a negative factor, thereby penalizing the network for learning representations that are strongly correlated with the sensitive attribute. This encourages the shared layers to develop representations that are less indicative of Gender, thus reducing bias.

Adversary Branch:
The network contains a separate branch (the adversary) that aims to predict the sensitive attribute (Gender) from the shared representation. The loss from the adversary is combined with the main loss (which predicts attrition), weighted by the hyperparameter λ. This combined objective pushes the shared representation to contain less information about Gender while still allowing accurate attrition predictions.

AMDN Preprocessing:
The model uses Adaptive Multi-Group Distribution Normalization (AMDN) as a preprocessing step. AMDN aligns feature distributions across different protected groups, reducing inherent biases in the input data. This results in a more balanced dataset that helps the model learn fairer representations.

Hyperparameter Tuning:
A series of hyperparameter configurations (varying learning rate, batch size, and λ) was tested to optimize model performance. The best trial was selected based on the F1 score, ensuring that the model achieves a balanced performance with acceptable trade-offs between precision and recall.

Performance Metrics
For the best trial (for example, with a learning rate of 0.0001, batch size of 32, and λ = 0.5), the model achieved:

Overall Accuracy: Approximately 82.31%

Recall: Approximately 59.57%

F1 Score: Approximately 51.85%

Balanced Accuracy: Approximately 73.11%

Precision: Approximately 45.90%

These metrics indicate that the model maintains robust overall performance while achieving a reasonable recall on the minority class, which is critical in scenarios where failing to detect attrition is costly.

Fairness Evaluation
Standard fairness metrics were computed using Fairlearn’s MetricFrame, which provides overall performance as well as subgroup-specific performance based on sensitive attributes (Gender and OverTime). The results revealed differences among subgroups:

For the subgroup with Gender=0 and OverTime=0, the model achieves high accuracy (around 91.67%) but moderate recall (42.86%).

For Gender=0 and OverTime=1, the model's accuracy drops to about 71.88% while recall improves to 66.67%.

For Gender=1 and OverTime=0, the model shows lower precision and recall (approximately 10% precision and 20% recall).

For Gender=1 and OverTime=1, the model performs strongly with approximately 77.27% accuracy, 73.91% precision, and 80.95% recall.

These subgroup differences suggest that while the adversarial model has reduced overall bias compared to earlier versions, some disparities still exist among specific groups.

Additional Fairness Metrics and Visualizations
To further quantify bias, we computed additional fairness metrics:

Demographic Parity Difference: Approximately 0.0864
This metric measures the difference in the positive prediction rates between the two gender groups. A smaller difference would indicate that both groups receive positive predictions at similar rates.

Equal Opportunity Difference: Approximately 0.0504
This metric measures the difference in true positive rates (recall) between groups. A lower value indicates that the model identifies the actual positive cases more equally across groups.

Average Odds Difference: Approximately 0.0618
This metric averages the differences in both true positive and false positive rates between groups, providing a combined measure of disparity. A value near 0 would suggest balanced error rates across groups.

Disparate Impact Ratio: Approximately 1.5568
This ratio compares the positive prediction rates between the groups. A ratio closer to 1 indicates parity, whereas a value significantly higher than 1 suggests that one group is much more likely to receive a positive prediction than the other.

Visualizations
Bar charts were generated to visualize the additional fairness metrics for the sensitive attribute Gender:

Demographic Parity per Group:
This chart shows the positive prediction rate for each gender group. The differences in the bars reflect the Demographic Parity Difference.

Equal Opportunity per Group:
This chart displays the true positive rate (recall) for each gender group, highlighting any differences in the model’s ability to correctly identify positives.

False Positive Rate per Group:
This chart shows the rate of false positives for each group, providing insight into the error distribution between genders.

These visualizations help illustrate the degree of disparity between groups and serve as a diagnostic tool to understand where further fairness improvements may be necessary.

Conclusion
The FCNN Adversarial Learning model, as implemented in train_fcnn_with_shap_for_new_preprocessing.py, demonstrates a significant improvement in reducing bias compared to earlier models. Although subgroup disparities remain (with certain groups, such as Gender=1 and OverTime=0, performing worse), the overall performance is robust, and the additional fairness metrics and visualizations provide valuable insights into the residual disparities. This model is currently the best candidate for ensuring a balanced trade-off between predictive accuracy and fairness across sensitive groups.

Detailed Comparison and Interpretation of FCNN with SHAP and FCNN Adversarial Learning Models
Overview
The two models under review are advanced approaches developed to improve fairness in predicting employee attrition. The FCNN with SHAP model focuses on explainability by integrating SHAP values to understand feature importance and uses advanced preprocessing (AMDN) to reduce bias. The FCNN Adversarial Learning model, on the other hand, incorporates an adversary branch with a gradient reversal layer to directly discourage the encoding of sensitive information (Gender) into the model's representations. Both models underwent hyperparameter tuning and were evaluated using overall performance metrics, subgroup-specific fairness metrics, and additional fairness measures. Visualizations (bar charts) were generated to provide an intuitive view of disparities between sensitive groups.

FCNN with SHAP Model
Performance Metrics
Overall Accuracy: ~82.31%

Recall: ~59.57%

F1 Score: ~51.85%

Balanced Accuracy: ~73.11%

Precision: ~45.90%

Interpretation:
The FCNN with SHAP model demonstrates moderate overall performance. The improved recall indicates that the model is better at detecting attrition cases compared to earlier baselines. However, the precision is relatively lower, suggesting that while more positive cases are captured, there may also be more false positives. The F1 score reflects this trade-off between precision and recall.

Standard Fairness Metrics (Fairlearn’s MetricFrame)
The model was evaluated on subgroups defined by Gender and OverTime:

Group (Gender=0, OverTime=0):

High accuracy (~91.67%) but moderate recall (~42.86%).

Group (Gender=0, OverTime=1):

Lower accuracy (~71.88%) with higher recall (~66.67%).

Group (Gender=1, OverTime=0):

Lower performance with very low precision (around 10%) and recall (approximately 20%).

Group (Gender=1, OverTime=1):

Strong performance with high recall (~80.95%) and high precision (~73.91%).

Interpretation:
The subgroup analysis shows that while the FCNN with SHAP model improves overall detection of attrition, significant disparities remain. In particular, the subgroup (Gender=1, OverTime=0) exhibits notably poor performance, indicating that the model may still be biased against that group.

Additional Fairness Metrics and Visualizations for FCNN with SHAP
Demographic Parity Difference: ~0.0864
(Indicates a moderate difference in the rate of positive predictions between gender groups.)

Equal Opportunity Difference: ~0.0504
(Shows a small gap in recall between the groups.)

Average Odds Difference: ~0.0618
(Reflects a moderate average difference in both true positive and false positive rates.)

Disparate Impact Ratio: ~1.5568
(Suggests that one gender is about 55.68% more likely to receive a positive prediction than the other.)

Visualization Interpretation:

The Demographic Parity chart displays a modest gap between the groups, supporting the numeric difference of 0.0864.

The Equal Opportunity chart shows a small difference in recall (about 0.0504), meaning that one group is slightly favored in correctly identifying positives.

The False Positive Rate chart reveals differences that, when averaged with TPR differences, result in an Average Odds Difference of ~0.0618.

The Disparate Impact Ratio indicates a notable imbalance, suggesting that further bias mitigation could help bring the ratio closer to 1.

FCNN Adversarial Learning Model
Performance Metrics
For the best trial (e.g., with lr=0.01, batch size=64, lambda_adv=1.0):

Overall Accuracy: ~86.05%

Recall: ~55.32%

F1 Score: ~55.91%

Precision: ~56.52%

Balanced Accuracy: ~73.61%

Interpretation:
The adversarial model achieves higher overall accuracy and F1 score compared to the FCNN with SHAP model. Although the recall is comparable, the improved precision in the adversarial model indicates a better balance between false positives and false negatives. This suggests that the adversarial training is effective at making the model less reliant on sensitive information while still performing well on the primary task.

Standard Fairness Metrics (Fairlearn’s MetricFrame)
Subgroup performance for the adversarial model:

Group (Gender=0, OverTime=0):

Accuracy ~90.48%, Recall ~42.86%

Group (Gender=0, OverTime=1):

Accuracy ~78.13%, Recall ~66.67%

Group (Gender=1, OverTime=0):

Accuracy ~85.07%, Recall ~20.00%

Group (Gender=1, OverTime=1):

Accuracy ~86.36%, Recall ~71.43%

Interpretation:
The adversarial model demonstrates a more balanced subgroup performance compared to the FCNN with SHAP model. While some disparities remain (for example, low recall in (Gender=1, OverTime=0)), the overall performance across groups is improved, reflecting the effectiveness of the gradient reversal and adversarial branch in reducing bias.

Additional Fairness Metrics and Visualizations for the Adversarial Model
Demographic Parity Difference: ~0.0164
(This very low value indicates nearly equal positive prediction rates across gender groups.)

Equal Opportunity Difference: ~-0.0141
(A value near zero suggests that both groups have almost identical true positive rates.)

Average Odds Difference: ~-0.0062
(Close to zero, indicating balanced error rates across groups.)

Disparate Impact Ratio: ~1.1117
(A ratio near 1 implies that positive predictions are almost evenly distributed between the groups.)

Visualization Interpretation:

The Demographic Parity chart shows nearly equal positive prediction rates, which aligns with the low difference of 0.0164.

The Equal Opportunity chart indicates almost balanced recall between the groups, with an Equal Opportunity Difference near zero.

The False Positive Rate chart demonstrates minimal disparity, contributing to the very low Average Odds Difference (-0.0062).

Overall Comparison and Conclusion
Performance Comparison:

The adversarial model outperforms the FCNN with SHAP model in overall accuracy (86.05% vs. ~82.31%) and achieves a higher F1 score (55.91% vs. ~51.85%).

The adversarial model also shows improved precision, leading to a more balanced trade-off between false positives and false negatives.

Fairness Comparison:

The additional fairness metrics for the adversarial model are significantly lower (or closer to zero) than those for the FCNN with SHAP model.

The Demographic Parity Difference and Equal Opportunity Difference for the adversarial model are near zero, indicating that positive predictions and recall are nearly equal across gender groups.

The Average Odds Difference is also almost zero, and the Disparate Impact Ratio is closer to 1, suggesting balanced treatment.

Visualizations Comparison:

Visualizations for the FCNN with SHAP model reveal moderate gaps between gender groups in positive prediction rates, TPR, and FPR.

In contrast, the adversarial model’s visualizations show nearly equal bar heights for all three metrics, supporting the numerical findings that the adversarial approach reduces bias more effectively.

Conclusion:
Based on both numerical metrics and visual evidence, the adversarial FCNN model (as implemented in train_fcnn_adversarial_learning.py) is superior in terms of bias reduction. It delivers higher overall performance while significantly reducing disparities across sensitive groups compared to the FCNN with SHAP model. This makes the adversarial model the best candidate when both predictive accuracy and fairness are prioritized.

Summary of Fairness Integration in Hyperparameter Tuning
We have updated our adversarial FCNN model's hyperparameter tuning process to explicitly incorporate fairness metrics into the model selection criteria. The changes are summarized as follows:

Fairness Penalty Calculation:

For each trial, in addition to calculating the standard F1 score, we compute additional fairness metrics for the sensitive attribute (Gender).

We derive a fairness penalty by summing the absolute differences in:

Positive prediction rates (Demographic Parity),

True Positive Rates (Equal Opportunity),

False Positive Rates.

Combined Score for Model Selection:

A combined score is calculated as:

Combined Score = F1 Score − β × (Fairness Penalty)

A fairness weight (β) is set to control the trade-off between performance and fairness. This ensures that the model selection process optimizes not only for high predictive performance but also for reduced bias.

Model Selection Based on Combined Score:

The best hyperparameter configuration is chosen based on the highest combined score, rather than solely on F1 score.

This approach favors models that maintain a good balance between overall accuracy and equitable treatment across sensitive groups.

By integrating these fairness metrics into the tuning process, our selected model is now more balanced in terms of both performance and fairness, which aligns with the ethical objectives of our project.

Below is a summary explaining the differences between the basic causal fairness analysis and the extended causal fairness analysis:

Causal Fairness Analysis

Builds a causal graph using DoWhy with Gender as the treatment and Attrition as the outcome, using other variables as common causes.

Identifies the causal effect and estimates it using methods like propensity score matching.

Runs one or two basic refutation tests (e.g., the placebo treatment refuter) to check the robustness of the estimated causal effect.

Creates a modified dataset by dropping Gender and trains a simple model on it to compare overall performance.

Extended Causal Fairness Analysis

Includes everything from the basic analysis and adds additional layers:

Incorporates multiple refutation tests (such as a data subset refuter, placebo treatment refuter, and (if applicable) others) for a more comprehensive validation of the causal effect.

Performs an intervention experiment by setting Gender to a constant value to see how that impacts the causal estimate.

Provides a more detailed comparison between the causal effect estimate and the model’s performance and fairness metrics when Gender is removed.

These extra steps help determine whether the observed bias is truly driven by Gender (i.e., causal) or simply a correlation.

This summary outlines how the extended analysis goes deeper by adding more refutation tests and intervention experiments, providing a richer understanding of whether Gender causally drives bias compared to the basic approach.

Causal Fairness Analysis Review
1. Causal Inference Using DoWhy
Causal Graph & Estimand:
We built a causal model using DoWhy, treating Gender as the treatment and Attrition as the outcome, with all other features as common causes. The identified estimand was of type NONPARAMETRIC_ATE, and the causal model was specified through a backdoor adjustment.

Estimated Causal Effect:
The estimated causal effect of Gender on Attrition is approximately 0.0463. This value suggests that, on average, a change in Gender is associated with a 4.63% change in the probability of Attrition.

Refutation Tests:
The Placebo Treatment Refuter was applied, resulting in a new effect of nearly zero (–0.00082) with a high p-value (0.96). This indicates that when a placebo treatment is used, the estimated effect essentially disappears. Such refutation results suggest that the observed effect of Gender may not be strongly causal but could be largely due to correlations.

2. Intervention Experiment
Intervention Setup:
We attempted an intervention experiment by setting all Gender values to a constant (0) to see if the causal effect could be re-estimated. However, the estimation failed because Gender became constant, which prevented NearestNeighbors from finding any valid samples. This highlights that variation in Gender is necessary for estimating its causal effect.

3. Modified Dataset Analysis (Dropping Gender)
Modified Dataset:
We created a modified dataset by dropping the Gender column entirely and retrained a simple FCNN model on this modified data.

Model Performance:
The model trained without Gender achieved:

Accuracy: 82.31%

Recall: 51.06%

F1 Score: 48.00%

These performance metrics are similar to the original model (which included Gender), indicating that removing Gender does not cause a dramatic degradation in overall performance.

4. Interpretation and Implications
Causal Effect Insight:
The causal analysis estimates a modest effect (4.63%) of Gender on Attrition. However, refutation tests—especially the placebo treatment refuter—suggest that this effect is not robust. In other words, when we simulate a placebo scenario, the effect nearly vanishes, implying that the influence of Gender may be driven by correlations rather than a strong causal relationship.

Impact of Removing Gender:
The modified model (without Gender) shows comparable performance (accuracy ~82.31%) with only a slight reduction in recall and F1 score. This minimal performance loss suggests that while Gender is correlated with Attrition, its removal does not severely impact predictive power. This finding supports the view that Gender may be contributing to bias, but its causal role is limited.

Overall Conclusion:
The combined evidence indicates that although Gender is associated with Attrition, its causal effect is modest and not robustly supported by the refutation tests. Moreover, the model trained without Gender maintains similar performance, suggesting that other factors might be influencing bias. These insights can help guide further bias mitigation strategies—possibly by focusing on other correlated features—rather than solely targeting Gender.

Extended Causal Fairness Analysis Review
Overview
The extended causal fairness analysis builds upon the basic causal fairness analysis by adding several key components that provide a more comprehensive understanding of how Gender influences Attrition. In the extended analysis, we:

Build a Detailed Causal Graph:
We define a causal model using DoWhy with Gender as the treatment, Attrition as the outcome, and all other relevant features as common causes. This graph clarifies the assumed relationships between variables.

Estimate the Causal Effect:
We identify and estimate the causal effect of Gender on Attrition using propensity score matching. The estimated effect is approximately 0.0463, meaning that, on average, a change in Gender is associated with a 4.63% change in Attrition probability.

Perform Multiple Refutation Tests:

Data Subset Refuter: Tests the robustness of the estimated effect on subsets of data.

Placebo Treatment Refuter: Applies a placebo treatment to see if the estimated effect disappears.
In our case, the placebo refuter yielded a near-zero effect (–0.00082) with a high p-value (0.96), suggesting that the observed effect is not robust.

Intervention Experiment:
An intervention is performed by setting all Gender values to a constant (0) to see if the causal effect can still be estimated. The estimation fails due to the constant treatment, reinforcing that variation in Gender is required for effect estimation.

Modified Dataset Analysis:
A separate analysis is conducted where Gender is dropped entirely from the dataset. A simple FCNN model trained on this modified dataset achieves performance metrics (Accuracy: 82.31%, Recall: 51.06%, F1 Score: 44.68%) that are comparable to the original model, indicating that removing Gender does not cause significant performance degradation.

Why the Extended Analysis is Superior
Richer Causal Insights:
By incorporating multiple refutation tests (data subset and placebo), the extended analysis provides stronger evidence on whether the observed effect of Gender is causal or merely a correlation. The near-zero effect in the placebo test strongly suggests that Gender’s causal influence is modest.

Intervention Experiment:
The intervention experiment (setting Gender to a constant) further challenges the model’s assumptions. Although the effect couldn’t be estimated due to the constant value, this failure itself provides insight: it confirms that variation in Gender is necessary to drive any effect, which indirectly supports the interpretation that Gender’s causal role is limited.

Comparative Performance Evaluation:
Training a model on a modified dataset (without Gender) and comparing its performance with the original model helps determine if removing Gender leads to improved fairness (or negligible performance loss). In this case, the modified model’s performance is very similar to that of the original, suggesting that Gender might not be a dominant driver of predictive power or bias.

Comprehensive Approach:
The extended analysis does not rely solely on a single causal effect estimate; it cross-validates the findings through refutation and intervention experiments. This comprehensive approach offers greater confidence in determining whether bias is causally driven by Gender or if other factors are at play.

Conclusion
The extended causal fairness analysis shows that:

The estimated causal effect of Gender on Attrition is modest (approximately 4.63%) and not robust to placebo tests.

An intervention experiment indicates that constant Gender values prevent causal estimation, underscoring the need for variability.

A model trained without Gender maintains similar performance, implying that removing Gender does not greatly harm predictive accuracy.

These findings together suggest that while Gender is associated with Attrition, its causal role in driving bias may be limited. This extended analysis is superior to a basic analysis because it adds multiple layers of validation, providing a deeper, more nuanced understanding of the relationships in the data. This, in turn, informs better bias mitigation strategies for the final model.

Detailed Interpretation of Interactive 3D Visualization Results For Train FCNN Adversarial Learning:

Points Represent Individual Trials:

Each point corresponds to a unique hyperparameter configuration (combination of learning rate, batch size, and λ) tested during the tuning process.

Hovering over a point in the interactive plot reveals the exact hyperparameter values and the resulting metrics (F1 score, fairness penalty, combined score).

Axes and Their Meaning:

X-axis (F1 Score):

Indicates overall predictive performance, with higher values representing a better balance of precision and recall.

Y-axis (Fairness Penalty):

Measures disparities in subgroup fairness metrics (e.g., differences in positive prediction rates, TPR, and FPR between sensitive groups). Lower values indicate more equitable predictions.

Z-axis (Combined Score):

Represents an integrated metric calculated as F1 score minus a fraction of the fairness penalty. Higher combined scores mean the model achieves a better overall trade-off between accuracy and fairness.

Evaluating Good vs. Poor Results:

High Combined Score (Good Results):

Points that are located high along the Z-axis (and appear in warm colors, such as yellow/orange) indicate configurations with a high F1 score and a low fairness penalty. These are considered optimal as they provide a balanced performance–fairness trade-off.

High F1 but High Fairness Penalty (Suboptimal):

Some points might exhibit a high F1 score yet also incur a large fairness penalty. This imbalance reduces the combined score, suggesting that despite good accuracy, the model may still be biased.

Low F1 with Low Fairness Penalty (Also Suboptimal):

Trials with very low F1 scores, even if they have low fairness penalties, result in low combined scores. These configurations indicate poor overall performance despite being fairer.

Overall Insights:

Trade-off Visualization:

The scatter plot visually demonstrates the inherent trade-off between maximizing predictive performance (F1 score) and minimizing bias (fairness penalty).

Hyperparameter Selection Guidance:

By identifying points with the highest combined scores, you can determine which hyperparameter settings offer the best balance. These “sweet spot” configurations indicate the most effective compromise between achieving high accuracy and reducing bias.

Practical Impact:

This interactive visualization empowers stakeholders to explore different model configurations and to understand how changes in hyperparameters affect both the accuracy and fairness of predictions. It ultimately supports informed decision-making for selecting the best model.

Detailed Interpretation of the 3D Scatter Plot Results
Each point in this interactive 3D scatter plot represents one hyperparameter trial of our adversarial FCNN model. In this visualization, the hyperparameters (learning rate, batch size, and adversarial weight λ) are varied, and each trial is evaluated on three key metrics:

X-axis: F1 Score
Indicates the model's overall performance by balancing precision and recall. A higher F1 score reflects better predictive performance.

Y-axis: Fairness Penalty
Measures disparities in subgroup metrics (such as differences in positive prediction rate, true positive rate, and false positive rate) between sensitive groups (here, Gender). A lower fairness penalty means that the model is more equitable.

Z-axis & Color Scale: Combined Score
Defined as:

Combined Score
=
F1 Score
−
𝛽
×
Fairness Penalty
Combined Score=F1 Score−β×Fairness Penalty
with β as a scaling factor (here, 0.5). A higher combined score indicates a better overall balance between performance and fairness.

What the Points Show
Good Results (High Combined Score):

Points with a high F1 score and a low fairness penalty will have a high combined score.

These points tend to appear toward the top (high Z-axis) and are color-coded in warm colors (orange/yellow).

Interpretation:

The model configuration corresponding to these points achieves strong predictive performance while keeping bias (as measured by subgroup differences) relatively low.

This indicates that the model is effective at reducing bias without sacrificing too much accuracy.

Poor Results (Low Combined Score):

Points with either a low F1 score, a high fairness penalty, or both will have a low combined score.

They appear lower on the Z-axis and are shown in cooler colors (blue/purple).

Interpretation:

Even if a model achieves a high F1 score, if it incurs a large fairness penalty (indicating significant bias between subgroups), the overall trade-off is poor.

Conversely, a model with a very low F1 score—even if it is fair—will still result in a low combined score.

These results suggest that the model configuration is either biased or underperforming, meaning further tuning or mitigation strategies are needed.

Trade-Off Analysis:

The scatter plot reveals the inherent tension between achieving high accuracy (F1 score) and low bias (fairness penalty).

Points that cluster with a good combined score indicate that certain hyperparameter settings allow the model to reach an optimal compromise.

This is essential, as it shows that it is possible to reduce bias (lower fairness penalty) without drastically compromising the F1 score.

What This Means for Bias Reduction
High Combined Scores:

When you observe points with high combined scores, it implies that the model is effective at both achieving high performance and reducing bias.

This is the ideal scenario for your project because it demonstrates that the adversarial approach (with proper tuning) can mitigate bias while still maintaining strong predictive power.

Low Combined Scores:

If the points fall into a region with low combined scores, it indicates that there is a significant trade-off issue—either the model is biased (high fairness penalty) or its predictive performance is poor (low F1), or both.

These configurations are less desirable and suggest that further refinement is necessary.

Interactive Exploration
The interactive aspect of the 3D visualization allows you to:

Hover Over Points:

View the specific hyperparameter values (learning rate, batch size, λ) that produced each result.

This detailed view helps identify which configurations yield a desirable balance.

Rotate and Zoom:

Explore the distribution of trials in 3D space to understand the overall landscape of performance and fairness trade-offs.

By analyzing this 3D visualization, you can conclude whether the model effectively reduces bias. If you see many points with high F1 scores and low fairness penalties (resulting in high combined scores), then the model is performing well on both fronts. Conversely, if the best points still have a relatively high fairness penalty, then you might need to further refine your approach or adjust the trade-off parameters. This detailed visual analysis provides a clear framework for selecting the optimal hyperparameter configuration for achieving both high accuracy and fairness.

