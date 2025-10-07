# titanic_preprocessing_xgbm_lgbm.py
import pandas as pd
import numpy as np
import os

# === PATH SETUP ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\15 XGBM & LGBM\XGBM & LGBM"
train_path = os.path.join(base_path, "Titanic_train.csv")
test_path = os.path.join(base_path, "Titanic_test.csv")

# === LOAD DATA ===
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("✅ Data Loaded Successfully")
print("Train Shape:", train_df.shape, "| Test Shape:", test_df.shape)

# ======================================================
# 1️⃣ IMPUTE MISSING VALUES
# ======================================================

# Check missing values
print("\n--- Missing Values Before Imputation ---")
print(train_df.isnull().sum())

# Fill Age with median (more robust to outliers)
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
test_df["Age"].fillna(train_df["Age"].median(), inplace=True)

# Fill Embarked with mode (most common value)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
test_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

# Fill Fare in test set with median
test_df["Fare"].fillna(train_df["Fare"].median(), inplace=True)

# Drop Cabin (too many missing)
train_df.drop(columns=["Cabin"], inplace=True, errors="ignore")
test_df.drop(columns=["Cabin"], inplace=True, errors="ignore")

# ======================================================
# 2️⃣ FEATURE ENGINEERING (OPTIONAL BUT USEFUL)
# ======================================================

# Extract Title from Name
for df in [train_df, test_df]:
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Mlle", "Ms", "Lady", "Countess", "Mme", "Dr", "Major", "Col", "Capt", "Sir", "Don", "Jonkheer", "Rev"],
        "Rare"
    )

# Family size
for df in [train_df, test_df]:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Drop non-useful columns
cols_to_drop = ["PassengerId", "Name", "Ticket"]
train_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
test_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# ======================================================
# 3️⃣ ENCODE CATEGORICAL VARIABLES
# ======================================================
# Columns like Sex, Embarked, Title, and Pclass need encoding
cat_cols = ["Sex", "Embarked", "Title", "Pclass"]

# One-hot encoding for categorical columns
train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

# Align test set to have same columns as train
train_cols = train_df.columns
test_df = test_df.reindex(columns=train_cols.drop("Survived"), fill_value=0)

# ======================================================
# 4️⃣ FINAL CLEANUP
# ======================================================
print("\n--- Missing Values After Imputation ---")
print(train_df.isnull().sum())

print("\n✅ Categorical Encoding Completed")
print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)

# ======================================================
# 5️⃣ SAVE CLEAN DATA
# ======================================================
train_clean_path = os.path.join(base_path, "Titanic_train_processed.csv")
test_clean_path = os.path.join(base_path, "Titanic_test_processed.csv")

train_df.to_csv(train_clean_path, index=False)
test_df.to_csv(test_clean_path, index=False)

print("\n✅ Preprocessing Completed Successfully!")
print("Processed files saved as:")
print(" -", train_clean_path)
print(" -", test_clean_path)
