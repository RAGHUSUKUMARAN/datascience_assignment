# eda_heart_disease.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === PATHS ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\13 decision tree\Decision Tree"
file_path = os.path.join(base_path, "heart_disease.xlsx")

# === LOAD DATA ===
df = pd.read_excel(file_path, sheet_name="Heart_disease")

# === BASIC INFO ===
print("Shape:", df.shape)
print("\n--- Info ---")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:\n", df.describe().T)

# === TARGET DISTRIBUTION ===
print("\nTarget distribution:\n", df['num'].value_counts())

# === HISTOGRAMS ===
numeric_cols = ['age','trestbps','chol','thalch','oldpeak']
df[numeric_cols].hist(bins=20, figsize=(12,8))
plt.suptitle("Histograms of Numeric Features")
plt.savefig(os.path.join(base_path, "histograms.png"))
plt.close()

# === BOXPLOTS ===
plt.figure(figsize=(12,8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2,3,i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "boxplots.png"))
plt.close()

# === CORRELATION MATRIX ===
corr = df[numeric_cols + ['num']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.savefig(os.path.join(base_path, "correlation_matrix.png"))
plt.close()

print("\nEDA completed. Plots saved in:", base_path)
print("- histograms.png")
print("- boxplots.png")
print("- correlation_matrix.png")
