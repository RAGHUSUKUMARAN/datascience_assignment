# Step 2: Data Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\16 KNN\KNN\Zoo.csv"
zoo_df = pd.read_csv(file_path)

# Drop non-numeric / irrelevant column
zoo_df = zoo_df.drop(columns=["animal name"])

# Check for outliers (mainly in 'legs')
plt.figure(figsize=(6,4))
sns.boxplot(x=zoo_df["legs"])
plt.title("Outlier Check — Legs Feature")
plt.show()

# Split features and target
X = zoo_df.drop(columns=["type"])
y = zoo_df["type"]

# Scale features (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
