# eda_glass_randomforest.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === PATH SETUP ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\14 random forest\Random Forest"
file_path = os.path.join(base_path, "glass.xlsx")

# === LOAD DATA ===
df = pd.read_excel(file_path, sheet_name="glass")

# === BASIC STRUCTURE ===
print("Shape of dataset:", df.shape)
print("\n--- Dataset Info ---")
print(df.info())

# === CHECK FOR MISSING VALUES ===
print("\n--- Missing Values ---")
print(df.isnull().sum())

# === CHECK FOR DUPLICATES ===
print("\nDuplicate rows:", df.duplicated().sum())

# === DESCRIPTIVE STATISTICS ===
print("\n--- Summary Statistics ---")
print(df.describe().T)

# === TARGET DISTRIBUTION ===
print("\n--- Target Value Counts (Type) ---")
print(df['Type'].value_counts())

# === HISTOGRAMS ===
num_cols = df.columns[:-1]  # all features except target
df[num_cols].hist(bins=15, figsize=(12, 8))
plt.suptitle("Feature Distributions - Glass Dataset", fontsize=14)
plt.savefig(os.path.join(base_path, "histograms.png"))
plt.close()

# === BOXPLOTS FOR OUTLIERS ===
plt.figure(figsize=(14, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "boxplots.png"))
plt.close()

# === CORRELATION MATRIX ===
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix - Glass Dataset")
plt.savefig(os.path.join(base_path, "correlation_matrix.png"))
plt.close()

# === PAIRPLOT (OPTIONAL - for visual patterns) ===
# sns.pairplot(df, hue='Type')
# plt.savefig(os.path.join(base_path, "pairplot.png"))
# plt.close()

print("\nEDA completed successfully.")
print("Plots saved in:", base_path)
print("Files created:")
print(" - histograms.png")
print(" - boxplots.png")
print(" - correlation_matrix.png")
# print(" - pairplot.png (optional, uncomment if needed)")
