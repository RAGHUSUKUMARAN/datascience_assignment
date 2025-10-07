import pandas as pd
from sklearn.ensemble import IsolationForest

# Load engineered dataset from Step 3
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2\adult_engineered.csv"
adult_df = pd.read_csv(file_path)

# Select numeric columns for outlier detection
num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain_log', 'capital_loss_log', 'hours_per_week']

# Isolation Forest
iso = IsolationForest(contamination=0.02, random_state=42)  # assume ~2% outliers
outlier_pred = iso.fit_predict(adult_df[num_cols])

# Keep only inliers
adult_cleaned = adult_df[outlier_pred == 1]

# Save cleaned dataset
out_path = r"D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2\adult_no_outliers.csv"
adult_cleaned.to_csv(out_path, index=False)

print("✅ Outlier detection complete. Cleaned dataset saved to:", out_path)
print("Original shape:", adult_df.shape, "→ After removing outliers:", adult_cleaned.shape)
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2\adult_encoded.csv"
adult_df = pd.read_csv(file_path)
