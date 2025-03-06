import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
df = pd.read_csv("data/processed_employee_attrition.csv")

# Set Seaborn style
sns.set_theme(style="whitegrid")

# 1️⃣ Plot Attrition Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Attrition"], hue=df["Attrition"], palette="coolwarm", legend=False)
plt.title("Attrition Distribution")
plt.xlabel("Attrition (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# 2️⃣ Check Correlation Between Features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# 3️⃣ Attrition by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Gender"], hue=df["Attrition"], palette="Set2")
plt.title("Attrition by Gender")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.legend(title="Attrition (0 = No, 1 = Yes)")
plt.show()

# 4️⃣ Attrition by OverTime
plt.figure(figsize=(6, 4))
sns.countplot(x=df["OverTime"], hue=df["Attrition"], palette="pastel")
plt.title("Attrition by OverTime")
plt.xlabel("OverTime (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title="Attrition (0 = No, 1 = Yes)")
plt.show()
