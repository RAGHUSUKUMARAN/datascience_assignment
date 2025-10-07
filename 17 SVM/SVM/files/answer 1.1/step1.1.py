# Step 1: Load and Explore the Mushroom Dataset

import pandas as pd

# Load dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\mushroom.csv"
mushroom_df = pd.read_csv(file_path)

# Basic exploration
print("Dataset shape:", mushroom_df.shape)
print("\nColumn names:\n", list(mushroom_df.columns))
print("\nData types:\n", mushroom_df.dtypes)
print("\nMissing values per column:\n", mushroom_df.isnull().sum())

# Preview dataset
print("\nFirst 5 rows:\n", mushroom_df.head())

# Descriptive summary (includes both numeric and categorical)
print("\nSummary statistics:\n", mushroom_df.describe(include='all'))
