# step3_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\mushroom.csv"
mushroom_df = pd.read_csv(file_path)

sns.set(style="whitegrid", palette="Set2")

# --- Step 1: Feature Distributions and Relationships ---

# Correct column names for UCI Mushroom dataset
selected_features = ['odor', 'spore_print_color', 'gill_color', 'cap_color', 'habitat']

# Sanity check for columns
print("Columns in dataset:", mushroom_df.columns.tolist())
for feature in selected_features:
    if feature not in mushroom_df.columns:
        print(f"⚠️ Warning: Column '{feature}' not found in dataset!")

plt.figure(figsize=(14, 10))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='class', data=mushroom_df)
    plt.title(f"Distribution of {feature} by Class")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\feature_distributions.png", dpi=150)
plt.show()

# --- Step 2: Class Distribution Visualization ---
plt.figure(figsize=(6, 5))
sns.countplot(x='class', data=mushroom_df, palette='Set1')
plt.title("Class Distribution (Edible vs. Poisonous)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig(r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\class_distribution.png", dpi=150)
plt.show()
