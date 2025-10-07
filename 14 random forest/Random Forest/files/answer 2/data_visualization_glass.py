# data_visualization_glass.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === PATH SETUP ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\14 random forest\Random Forest"
file_path = os.path.join(base_path, "glass.xlsx")

# === LOAD DATA ===
df = pd.read_excel(file_path, sheet_name="glass")

# === BASIC INFO ===
print("Shape of dataset:", df.shape)
print("Columns:", list(df.columns))
print("\nGlass Types:", df['Type'].unique())

# Separate features and target
features = df.columns[:-1]
target = 'Type'

# === HISTOGRAMS ===
plt.figure(figsize=(12, 8))
df[features].hist(bins=15, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions - Glass Dataset", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "histograms.png"))
plt.close()

# === BOXPLOTS ===
plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col], color='lightcoral')
    plt.title(col, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "boxplots.png"))
plt.close()

# === PAIRPLOT (Visualizing Feature Relationships by Glass Type) ===
sns.pairplot(df, hue='Type', palette='husl', diag_kind='hist')
plt.suptitle("Pairplot - Relationships Between Features and Glass Type", y=1.02, fontsize=14)
plt.savefig(os.path.join(base_path, "pairplot.png"))
plt.close()

# === CORRELATION HEATMAP ===
plt.figure(figsize=(10, 8))
corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Glass Dataset", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "correlation_heatmap.png"))
plt.close()

print("\n✅ Data Visualization Completed.")
print("Visual files saved in:", base_path)
print(" - histograms.png")
print(" - boxplots.png")
print(" - pairplot.png")
print(" - correlation_heatmap.png")

# === BASIC INSIGHTS ===
print("\n--- ANALYSIS INSIGHTS ---")
print("1. Some features like 'K', 'Ba', and 'Fe' have strong skew (many zeros).")
print("2. 'RI', 'Na', 'Mg', and 'Ca' show wider variation across glass types.")
print("3. Pairplot reveals partial separation between certain glass types using 'Mg' and 'Al'.")
print("4. Correlation heatmap shows:")
print("   - 'Al' and 'Mg' are negatively correlated.")
print("   - 'RI' and 'Si' show mild inverse relationship.")
print("   - 'Ca' correlates positively with 'RI' and negatively with 'Al'.")
print("\nThese patterns will help Random Forest identify which elements drive glass classification.")
