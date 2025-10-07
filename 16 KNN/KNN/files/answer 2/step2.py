# Step 2: Data Preprocessing for Zoo dataset
# - Handles missing values check
# - Visualizes possible outliers (legs)
# - Drops irrelevant columns
# - Scales features for KNN
# - Splits data (80% train / 20% test) with stratification

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib   # optional: to save the scaler

# -------------------------
# 1. Load dataset (try user path, fallback to /mnt/data)
# -------------------------
user_path = r"D:\DATA SCIENCE\ASSIGNMENTS\16 KNN\KNN\Zoo.csv"
fallback_path = "/mnt/data/Zoo.csv"

if os.path.exists(user_path):
    file_path = user_path
elif os.path.exists(fallback_path):
    file_path = fallback_path
else:
    raise FileNotFoundError(
        f"Zoo.csv not found at either '{user_path}' or '{fallback_path}'. "
        "Put the file in one of these paths or update file_path variable."
    )

df = pd.read_csv(file_path)
print(f"Loaded file: {file_path}")
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows:\n", df.head())

# -------------------------
# 2. Missing values check
# -------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# -------------------------
# 3. Drop irrelevant columns
# -------------------------
if "animal name" in df.columns:
    df = df.drop(columns=["animal name"])
    print("\nDropped column: 'animal name'")

# -------------------------
# 4. Quick stats & outlier check for 'legs'
# -------------------------
print("\nSummary statistics for numeric features:\n", df.describe())

plt.figure(figsize=(6, 3))
sns.boxplot(x=df["legs"])
plt.title("Boxplot — legs")
plt.xlabel("Number of legs")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x="legs", data=df)
plt.title("Countplot — legs distribution")
plt.xlabel("Legs")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# NOTE: legs values like 0 and 8 are biologically valid (snakes, arachnids), so we keep them.

# -------------------------
# 5. Split features and target
# -------------------------
if "type" not in df.columns:
    raise KeyError("'type' column (target) not found in dataset. Make sure file contains target column named 'type'.")

X = df.drop(columns=["type"])
y = df["type"]

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print("\nTarget class distribution:\n", y.value_counts().sort_index())

# -------------------------
# 6. Feature scaling (StandardScaler)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # X is mostly binary + 'legs', so standardization is appropriate

# Optional: save scaler for later (useful when deploying)
scaler_outpath = os.path.join(os.path.dirname(file_path), "zoo_scaler.joblib")
joblib.dump(scaler, scaler_outpath)
print(f"\nStandardScaler saved to: {scaler_outpath}")

# -------------------------
# 7. Train-test split (80% train, 20% test) with stratification
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nAfter splitting:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train distribution:\n", pd.Series(y_train).value_counts().sort_index())
print("y_test distribution:\n", pd.Series(y_test).value_counts().sort_index())

# -------------------------
# 8. (Optional) Save train/test as .npz or .csv for next steps
# -------------------------
out_dir = os.path.dirname(file_path)
import numpy as np
np.savez_compressed(os.path.join(out_dir, "zoo_knn_data.npz"),
                    X_train=X_train, X_test=X_test, y_train=y_train.values, y_test=y_test.values)
print(f"\nSaved processed arrays to: {os.path.join(out_dir, 'zoo_knn_data.npz')}")

# End of Step 2 preprocessing script
