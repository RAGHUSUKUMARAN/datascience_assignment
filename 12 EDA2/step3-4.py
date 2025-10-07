"""
full_outlier_pipeline.py

Usage:
    python full_outlier_pipeline.py

What it does:
- Looks for adult_engineered.csv in your folder. If missing:
    - tries to load adult_encoded.csv and add missing engineered columns
    - if that is missing, tries to load the raw adult_with_headers.csv and runs basic encoding+engineering
- Performs IsolationForest outlier detection on selected numeric columns
- Saves adult_engineered.csv and adult_no_outliers.csv to the folder:
    D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2\
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# === CONFIG: change if your folder is different ===
BASE_DIR = r"D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2"
ENGINEERED_PATH = os.path.join(BASE_DIR, "adult_engineered.csv")
ENCODED_PATH = os.path.join(BASE_DIR, "adult_encoded.csv")
RAW_PATH     = os.path.join(BASE_DIR, "adult_with_headers.csv")
OUTLIER_PATH = os.path.join(BASE_DIR, "adult_no_outliers.csv")
# ==================================================

def load_csv_safe(path):
    if os.path.exists(path):
        print(f"Loading: {path}")
        return pd.read_csv(path)
    return None

def basic_encoding_from_raw(df):
    """
    Minimal encoding of raw Adult dataset:
    - Replace '?' -> NaN, fill with 'Unknown' for categoricals
    - One-hot encode sex and race (as per instructions)
    - Label-encode other categorical cols with >5 categories
    - Map target income to 0/1
    Returns encoded dataframe
    """
    print("Performing basic encoding from raw dataset...")
    df = df.copy()
    # normalize whitespace and replace '?' with NaN
    df = df.replace(' ?', np.nan)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # fill missing categorical with 'Unknown'
    cat_cols = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown')
    # One-hot encode sex and race
    for c in ['sex','race']:
        if c in df.columns:
            # keep all columns (do not drop first); easier to track
            df = pd.get_dummies(df, columns=[c], prefix=c)
    # Label encode other categorical features (except ones just one-hot encoded and target)
    label_cols = [c for c in cat_cols if c in df.columns and not c.startswith('sex') and not c.startswith('race')]
    # Remove 'sex' and 'race' if they are now one-hot columns names
    label_cols = [c for c in label_cols if c not in ['sex','race']]
    le = LabelEncoder()
    for c in label_cols:
        # convert to str to ensure consistent encoding
        df[c] = le.fit_transform(df[c].astype(str))
    # target encode
    if 'income' in df.columns:
        df['income'] = df['income'].map({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1})
    return df

def ensure_engineered(df):
    """
    Ensure the engineered features exist:
    - age_group (Young, Middle-aged, Senior)
    - work_hours_cat (Part-time, Full-time, Over-time, Heavy Workload)
    - capital_gain_log, capital_loss_log
    If these cols exist already, function will not overwrite them.
    """
    df = df.copy()
    # Age group
    if 'age_group' not in df.columns:
        print("Creating 'age_group'...")
        df['age_group'] = pd.cut(df['age'], bins=[0,30,50,200], labels=['Young','Middle-aged','Senior'])
    else:
        print("'age_group' exists; skipping creation.")
    # Work hours category
    if 'work_hours_cat' not in df.columns:
        print("Creating 'work_hours_cat'...")
        df['work_hours_cat'] = pd.cut(df['hours_per_week'], bins=[0,30,40,60,200], labels=['Part-time','Full-time','Over-time','Heavy Workload'])
    else:
        print("'work_hours_cat' exists; skipping creation.")
    # Log transforms for capital gain/loss
    if 'capital_gain_log' not in df.columns:
        if 'capital_gain' in df.columns:
            print("Creating 'capital_gain_log'...")
            df['capital_gain_log'] = np.log1p(df['capital_gain'].fillna(0).astype(float))
        else:
            print("Warning: 'capital_gain' not found; skipping capital_gain_log.")
    else:
        print("'capital_gain_log' exists; skipping creation.")
    if 'capital_loss_log' not in df.columns:
        if 'capital_loss' in df.columns:
            print("Creating 'capital_loss_log'...")
            df['capital_loss_log'] = np.log1p(df['capital_loss'].fillna(0).astype(float))
        else:
            print("Warning: 'capital_loss' not found; skipping capital_loss_log.")
    else:
        print("'capital_loss_log' exists; skipping creation.")
    return df

def main():
    # 1) Try engineered first
    df = load_csv_safe(ENGINEERED_PATH)
    if df is not None:
        print("Found engineered dataset. Verifying engineered columns...")
        df = ensure_engineered(df)
    else:
        # 2) Try encoded
        df = load_csv_safe(ENCODED_PATH)
        if df is not None:
            print("Found encoded dataset. Adding engineered columns if missing...")
            df = ensure_engineered(df)
        else:
            # 3) Try raw
            df = load_csv_safe(RAW_PATH)
            if df is not None:
                print("Found raw dataset. Running basic encoding + feature engineering...")
                df = basic_encoding_from_raw(df)
                df = ensure_engineered(df)
            else:
                print("ERROR: None of the expected files were found in the folder:")
                print(f"  - {ENGINEERED_PATH}")
                print(f"  - {ENCODED_PATH}")
                print(f"  - {RAW_PATH}")
                print("Place one of these files in the folder and re-run.")
                sys.exit(1)

    # Save engineered dataset (overwrite or create)
    try:
        df.to_csv(ENGINEERED_PATH, index=False)
        print(f"Saved engineered dataset to: {ENGINEERED_PATH}")
    except Exception as e:
        print("Failed to save engineered CSV:", e)
        sys.exit(1)

    # --------------- Isolation Forest Outlier Detection ---------------
    # Prepare numeric cols for outlier detection; if log columns are present prefer them
    numeric_candidates = []
    # prefer log versions if available
    for c in ['age','fnlwgt','education_num','capital_gain_log','capital_loss_log','hours_per_week']:
        if c in df.columns:
            numeric_candidates.append(c)
    if not numeric_candidates:
        print("No numeric columns found for outlier detection. Exiting.")
        sys.exit(1)
    print("Numeric columns used for outlier detection:", numeric_candidates)

    # Drop rows with NaN in numeric candidates (IsolationForest cannot handle NaN)
    df_num = df[numeric_candidates].copy()
    nan_count = df_num.isnull().any(axis=1).sum()
    if nan_count > 0:
        print(f"Found {nan_count} rows with NaN in numeric columns; dropping those rows for outlier detection.")
        valid_idx = ~df_num.isnull().any(axis=1)
        df_for_iso = df.loc[valid_idx, :].copy()
    else:
        df_for_iso = df.copy()

    X = df_for_iso[numeric_candidates].astype(float).values

    # Configure Isolation Forest
    contamination = 0.02  # you can adjust this; 0.02 means ~2% anomalies
    iso = IsolationForest(contamination=contamination, random_state=42)
    print("Fitting IsolationForest...")
    iso.fit(X)
    preds = iso.predict(X)  # -1 for outlier, 1 for inlier

    # Keep only inliers
    inlier_mask = (preds == 1)
    df_inliers = df_for_iso.loc[inlier_mask, :].copy()

    # If we dropped rows earlier due to NaN, combine with rows that were excluded (we choose to exclude them from final dataset too)
    final_df = df_inliers.copy()

    # Save outlier-removed dataset
    try:
        final_df.to_csv(OUTLIER_PATH, index=False)
        print(f"Saved outlier-removed dataset to: {OUTLIER_PATH}")
    except Exception as e:
        print("Failed to save outlier-removed CSV:", e)
        sys.exit(1)

    print("Original rows:", len(df), "→ After removing outliers:", len(final_df))
    print("Done ✅")

if __name__ == "__main__":
    main()
