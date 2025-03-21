import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv(r"C:\Users\Sulaiman Mahmood\Documents\Employee_Retension_Fairness_Project\Dataset Folder\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Drop unnecessary columns
columns_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ["Attrition", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# Normalize numerical columns
scaler = MinMaxScaler()
numerical_columns = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", "MonthlyRate",
                     "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
                     "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display cleaned dataset info
print("Cleaned Dataset Info:")
print(df.info())

# Display first few rows
print("\nFirst 5 Rows of Cleaned Dataset:")
print(df.head())

# Save preprocessed data for future use
df.to_csv("processed_employee_attrition.csv", index=False)
print("\n✅ Preprocessed dataset saved as 'processed_employee_attrition.csv'")
