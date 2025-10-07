# feature_engineering_heart.py
import pandas as pd
import numpy as np
import os

# === PATHS ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\13 decision tree\Decision Tree"
file_path = os.path.join(base_path, "heart_disease.xlsx")

# === LOAD DATA ===
df = pd.read_excel(file_path, sheet_name="Heart_disease")

# === HANDLE MISSING VALUES ===
# Replace missing oldpeak values with median
df['oldpeak'] = df['oldpeak'].fillna(df['oldpeak'].median())

# === FIX ANOMALIES ===
# Replace 0 in trestbps and chol with median (clinically impossible values)
for col in ['trestbps', 'chol']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# === CONVERT CATEGORICALS ===
# Sex → binary
df['sex'] = df['sex'].map({'Male':1, 'Female':0})

# Boolean columns (fbs, exang) → integers
for col in ['fbs','exang']:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
    elif df[col].dtype == object:   # handle weird cases like 'TURE', 'FALSE'
        df[col] = df[col].map({'True':1, 'TURE':1, 'False':0, 'FALSE':0}).fillna(0).astype(int)

# Target → binary (disease present or not)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# One-hot encode categorical features
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop original "num"
df_processed = df_processed.drop(columns=['num'])

# === SAVE PROCESSED DATASET ===
processed_path = os.path.join(base_path, "heart_processed.csv")
df_processed.to_csv(processed_path, index=False)

print("Feature Engineering completed.")
print("Processed dataset saved to:", processed_path)
print("Final shape:", df_processed.shape)
print("Columns:", df_processed.columns.tolist()[:10], "...")  # preview
