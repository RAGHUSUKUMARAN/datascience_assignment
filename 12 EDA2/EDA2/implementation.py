import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2\adult_with_headers.csv"
adult_df = pd.read_csv(file_path)

# Replace '?' with NaN for consistency
adult_df = adult_df.replace("?", None)

# One-Hot Encoding for features with < 5 categories
adult_df = pd.get_dummies(adult_df, columns=["sex", "race"], drop_first=True)

# Label Encoding for features with > 5 categories
label_enc = LabelEncoder()
for col in ["workclass", "education", "marital_status", "occupation", "relationship", "native_country"]:
    adult_df[col] = label_enc.fit_transform(adult_df[col].astype(str))

# Encode target (income: <=50K=0, >50K=1)
adult_df["income"] = adult_df["income"].map({"<=50K": 0, ">50K": 1})

# Save encoded dataset
out_path = r"D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2\adult_encoded.csv"
adult_df.to_csv(out_path, index=False)

print("✅ Encoding complete. Encoded dataset saved to:", out_path)