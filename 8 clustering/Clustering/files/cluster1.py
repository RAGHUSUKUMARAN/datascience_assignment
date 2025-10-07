import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your dataset (from the "data" sheet)
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\8 clustering\Clustering\EastWestAirlines.csv"
df = pd.read_csv(r"D:\DATA SCIENCE\ASSIGNMENTS\8 clustering\Clustering\EastWestAirlines.csv")


# Drop ID# column if present
if "ID#" in df.columns:
    df = df.drop(columns=["ID#"])

print("Original shape:", df.shape)

# Step 1: Handle missing values (none in this dataset, but good to keep)
print("\nMissing values per column:\n", df.isna().sum())

# Step 2: Outlier removal using IQR
def remove_outliers_iqr(data):
    df_out = data.copy()
    for col in df_out.select_dtypes(include=[np.number]).columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
    return df_out

df_no_outliers = remove_outliers_iqr(df)
print("\nAfter outlier removal:", df_no_outliers.shape)

# Step 3: Scaling numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_no_outliers.drop(columns=["Award?"]))
df_scaled = pd.DataFrame(scaled_data, columns=df_no_outliers.drop(columns=["Award?"]).columns)

# Keep Award? column separately (optional for clustering)
df_scaled["Award?"] = df_no_outliers["Award?"].values

# Save preprocessed dataset to the same folder
output_path = r"D:\DATA SCIENCE\ASSIGNMENTS\8 clustering\Clustering\EastWestAirlines_Preprocessed.xlsx"
df_scaled.to_excel(output_path, index=False)

print(f"\nPreprocessed dataset saved to: {output_path}")
