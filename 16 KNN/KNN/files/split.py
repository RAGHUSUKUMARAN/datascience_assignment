# Step 3: Splitting the dataset into training and testing sets (80–20)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\16 KNN\KNN\Zoo.csv"
zoo_df = pd.read_csv(file_path)

# Drop irrelevant column
zoo_df = zoo_df.drop(columns=["animal name"])

# Split features and target
X = zoo_df.drop(columns=["type"])
y = zoo_df["type"]

# Standardize features (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset — 80% train, 20% test with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("\nClass distribution in Training set:\n", y_train.value_counts(normalize=True).round(2))
print("\nClass distribution in Testing set:\n", y_test.value_counts(normalize=True).round(2))
