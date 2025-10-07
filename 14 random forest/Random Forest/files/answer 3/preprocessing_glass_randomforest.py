# preprocessing_glass_randomforest_safe.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import warnings

# === PATH SETUP ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\14 random forest\Random Forest"
file_path = os.path.join(base_path, "glass.xlsx")
processed_path = os.path.join(base_path, "glass_processed.csv")

# === LOAD DATA ===
df = pd.read_excel(file_path, sheet_name="glass")

print("✅ Dataset loaded successfully.")
print("Shape:", df.shape)
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# =========================================================
# 1️⃣ HANDLE MISSING VALUES
# =========================================================
if df.isnull().sum().sum() == 0:
    print("\nNo missing values found — no imputation needed.")
else:
    print("\nMissing values detected. Applying median imputation.")
    df = df.fillna(df.median())

# =========================================================
# 2️⃣ ENCODE CATEGORICAL VARIABLES
# =========================================================
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    print("\nCategorical columns found:", list(cat_cols))
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
else:
    print("\nNo categorical columns — encoding not required.")

# =========================================================
# 3️⃣ FEATURE SCALING
# =========================================================
X = df.drop(columns=['Type'])
y = df['Type']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("\nFeature scaling (Standardization) applied successfully.")
print("Mean of scaled features (approx):\n", X_scaled.mean().round(3))
print("Std dev of scaled features (approx):\n", X_scaled.std().round(3))

# =========================================================
# 4️⃣ HANDLE IMBALANCED DATA (try SMOTE, else fallback)
# =========================================================
print("\n--- Target Distribution Before Balancing ---")
print(y.value_counts())

use_smote = False
try:
    from imblearn.over_sampling import SMOTE
    use_smote = True
except Exception as e:
    print("\nNote: imbalanced-learn / SMOTE not available or failed to import.")
    print("Reason:", str(e))
    print("Proceeding without SMOTE (data will remain unbalanced).")

if use_smote:
    # Try SMOTE, but protect against runtime errors (e.g., too few samples for k_neighbors)
    try:
        # For small classes, set k_neighbors to min(3, n_min_class-1)
        from collections import Counter
        class_counts = Counter(y)
        n_min = min(class_counts.values())
        k_neighbors = 3
        if n_min <= 3:
            k_neighbors = max(1, n_min - 1)  # SMOTE requires k_neighbors < n_min, adjust down safely
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_bal, y_bal = smote.fit_resample(X_scaled, y)
        print("\nSMOTE applied successfully.")
        print("--- Target Distribution After SMOTE Balancing ---")
        print(pd.Series(y_bal).value_counts())
        final_X, final_y = X_bal, y_bal
    except Exception as e:
        warnings.warn(f"SMOTE failed at fit_resample: {e}. Proceeding without SMOTE.")
        final_X, final_y = X_scaled, y
else:
    final_X, final_y = X_scaled, y

# =========================================================
# SAVE PROCESSED DATA
# =========================================================
processed_df = pd.concat([pd.DataFrame(final_X, columns=X.columns), pd.Series(final_y, name="Type")], axis=1)
processed_df.to_csv(processed_path, index=False)

print(f"\n✅ Data Preprocessing Completed Successfully.")
print("Processed file saved as:", processed_path)
print("Final shape:", processed_df.shape)
