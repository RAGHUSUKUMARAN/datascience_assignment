# step2_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the encoded dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\mushroom.csv"
mushroom_df = pd.read_csv(file_path)

# Display shape and first few rows
print("Initial dataset shape:", mushroom_df.shape)
print(mushroom_df.head())

# --- Step 1: Encode Categorical Variables ---

# Separate features and target
X = mushroom_df.drop('class', axis=1)
y = mushroom_df['class']

# Encode target label (edible/poisonous)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Perform one-hot encoding for categorical predictors
X_encoded = pd.get_dummies(X, drop_first=True)

print("After encoding:")
print("Feature matrix shape:", X_encoded.shape)
print("Target vector shape:", y_encoded.shape)

# --- Step 2: Split Dataset ---

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# --- Save preprocessed data ---
X_train.to_csv(r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\X_train.csv", index=False)
X_test.to_csv(r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\X_test.csv", index=False)
pd.DataFrame(y_train, columns=['class']).to_csv(r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\y_train.csv", index=False)
pd.DataFrame(y_test, columns=['class']).to_csv(r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\y_test.csv", index=False)

print("Preprocessing complete. Encoded and split data saved successfully.")
