import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\9 PCA\PCA\wine.csv"
df = pd.read_csv(file_path)

# Basic exploration
print("Dataset shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nFirst 5 rows:\n", df.head())

# Histograms
df.drop(columns=['Type']).hist(bins=20, figsize=(15,10))
plt.suptitle("Histograms of Features", y=1.02)
plt.show()

# Boxplots
plt.figure(figsize=(15,8))
sns.boxplot(data=df.drop(columns=['Type']), orient="h")
plt.title("Boxplots of Features")
plt.show()

# Density plots for first 5 features
num_cols = df.drop(columns=['Type']).columns
plt.figure(figsize=(12,8))
for col in num_cols[:5]:
    sns.kdeplot(x=df[col], label=col, fill=True, alpha=0.3)
plt.title("Density plots (first 5 features)")
plt.legend()
plt.show()

# Correlation heatmap
corr = df.drop(columns=['Type']).corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", annot=False, cbar=True)
plt.title("Correlation Heatmap of Features")
plt.show()
