import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\9 PCA\PCA\wine.csv"
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(columns=['Type'])
y = df['Type']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_var = pca.explained_variance_ratio_
cumulative_var = explained_var.cumsum()

# Scree plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_var)+1), explained_var, marker='o', label="Individual")
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='s', label="Cumulative")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot & Cumulative Explained Variance")
plt.legend()
plt.show()

# Transform dataset to first 2 principal components
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

# Plot 2D PCA with wine classes
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca2[:,0], y=X_pca2[:,1], hue=y, palette="Set1", alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine Dataset projected onto first 2 Principal Components")
plt.show()
