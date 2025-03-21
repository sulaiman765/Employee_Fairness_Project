import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def adaptive_multi_group_distribution_normalization(
    df,
    protected_attr,
    target_col=None,
    n_clusters=3,
    reference="global",
    resample=True,
    random_state=42
):
    """
    Perform the AMDN procedure:
      1) Split data by protected groups
      2) KMeans-cluster each group
      3) Align each group's cluster distribution to a reference
      4) (Optionally) oversample smaller groups/clusters
    """
    # Separate target if provided
    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df
        y = None

    # Identify numeric columns to cluster/normalize
    # We'll skip the protected attribute in clustering
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if protected_attr in numeric_cols:
        numeric_cols.remove(protected_attr)

    # ============= STEP 1: Group Separation ================
    groups = X[protected_attr].unique()

    # We'll store the transformed data here
    X_transformed_list = []
    y_transformed_list = []

    # (Optional) Precompute reference stats if reference='global'
    if reference == "global":
        # Compute global mean/std for numeric columns across entire dataset
        global_mean = X[numeric_cols].mean()
        global_std = X[numeric_cols].std(ddof=0)  # population-like std
    else:
        global_mean, global_std = None, None

    # ============= STEP 2 & 3: Cluster Each Group & Normalize =============
    for g in groups:
        # Subset for group g
        df_g = X[X[protected_attr] == g].copy()
        if y is not None:
            y_g = y.loc[df_g.index]
        else:
            y_g = None

        # Cluster on numeric features (excluding protected_attr)
        cluster_data = df_g[numeric_cols]

        # If there's not enough data to do KMeans with n_clusters,
        # fallback to a smaller number
        if len(cluster_data) < n_clusters:
            n_used = 1
        else:
            n_used = n_clusters

        kmeans = KMeans(n_clusters=n_used, random_state=random_state)
        cluster_labels = kmeans.fit_predict(cluster_data)

        # For each cluster, compute mean & std, then align
        new_rows = []
        new_targets = []

        for cluster_id in range(n_used):
            cluster_mask = (cluster_labels == cluster_id)
            S_gk = df_g.loc[cluster_mask, numeric_cols]

            if len(S_gk) == 0:
                continue

            mu_gk = S_gk.mean()
            sigma_gk = S_gk.std(ddof=0)

            # Determine reference stats
            if reference == "global":
                mu_ref = global_mean
                sigma_ref = global_std
            elif reference == "privileged":
                # Placeholder for a privileged-group reference
                mu_ref = global_mean
                sigma_ref = global_std
            else:
                raise ValueError("Unknown reference mode. Use 'global' or 'privileged'.")

            # Avoid division by zero
            sigma_replace = sigma_gk.replace({0: 1e-8})

            # Normalization
            #   X' = (X - mu_gk) * (sigma_ref / sigma_gk) + mu_ref
            S_gk_normalized = (S_gk - mu_gk) * (sigma_ref / sigma_replace) + mu_ref

            # Replace numeric columns with normalized version
            df_gk_norm = df_g.loc[cluster_mask].copy()
            df_gk_norm[numeric_cols] = S_gk_normalized

            new_rows.append(df_gk_norm)
            if y_g is not None:
                new_targets.append(y_g.loc[df_gk_norm.index])

        # Combine all clusters for group g
        g_transformed = pd.concat(new_rows, axis=0)
        if y_g is not None:
            g_targets = pd.concat(new_targets, axis=0)
        else:
            g_targets = None

        # We handle oversampling later
        X_transformed_list.append(g_transformed)
        if g_targets is not None:
            y_transformed_list.append(g_targets)

    # Concat all groups
    X_amdn = pd.concat(X_transformed_list, axis=0)
    if y is not None:
        y_amdn = pd.concat(y_transformed_list, axis=0)
    else:
        y_amdn = None

    # ============= STEP 4: (Optional) Oversampling at Group Level =============
    if resample:
        final_df = X_amdn.copy()
        if y_amdn is not None:
            final_df[target_col] = y_amdn

        # Group by the protected attribute
        group_sizes = final_df.groupby(protected_attr).size()
        max_size = group_sizes.max()

        oversampled_dfs = []
        for g, df_sub in final_df.groupby(protected_attr):
            size_g = len(df_sub)
            if size_g < max_size:
                df_over = df_sub.sample(n=max_size, replace=True, random_state=random_state)
            else:
                df_over = df_sub
            oversampled_dfs.append(df_over)

        final_oversampled = pd.concat(oversampled_dfs, axis=0).sample(frac=1.0, random_state=random_state)

        if y_amdn is not None:
            y_amdn = final_oversampled[target_col]
            X_amdn = final_oversampled.drop(columns=[target_col])
        else:
            X_amdn = final_oversampled

    return X_amdn, y_amdn


if __name__ == "__main__":
    # 1. LOAD DATA
    file_path = r"C:\Users\Sulaiman Mahmood\Documents\Employee_Retension_Fairness_Project\Dataset Folder\WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(file_path)

    # 2. DROP UNUSED COLUMNS
    columns_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    # 3. LABEL-ENCODE CATEGORICAL COLUMNS
    #    (including Attrition => numeric, Gender => numeric)
    label_encoders = {}
    categorical_columns = [
        "Attrition", "BusinessTravel", "Department", "EducationField",
        "Gender", "JobRole", "MaritalStatus", "OverTime"
    ]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 4. MIN-MAX SCALE NUMERICAL COLUMNS
    scaler = MinMaxScaler()
    numerical_columns = [
        "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
        "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
        "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # 5. SPLIT INTO TRAIN & TEST (AVOID DATA LEAKAGE)
    target_col = "Attrition"
    protected_attr = "Gender"

    # Stratify by both target + protected attribute, if desired
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[[target_col, protected_attr]]
    )

    # 6. APPLY THE AMDN PROCEDURE (TO TRAIN DATA ONLY)
    X_train_amdn, y_train_amdn = adaptive_multi_group_distribution_normalization(
        df=train_df,
        protected_attr=protected_attr,
        target_col=target_col,
        n_clusters=3,         # how many clusters per group
        reference="global",   # align to global distribution
        resample=True,
        random_state=42
    )

    # 7. SAVE FINAL TRAIN DATA
    train_processed = X_train_amdn.copy()
    train_processed[target_col] = y_train_amdn
    train_processed.to_csv("train_amdn.csv", index=False)

    print("✅ AMDN preprocessing complete! Created 'train_amdn.csv'")

    # NOTE: For the test set, you typically do not re-cluster or re-learn distributions.
    #       You can either:
    #       1) Use the same scaling/encoding from your train set (which you already applied above).
    #       2) Possibly apply partial alignment from the train clusters (more advanced).
    #       For now, test_df remains as-is (already label-encoded & scaled).
