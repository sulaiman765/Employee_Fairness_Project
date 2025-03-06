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
