import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\9 PCA\PCA\wine.csv"
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(columns=['Type'])
y = df['Type']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (reduce to 2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Try different k values on PCA data
inertia = []
sil_scores = []
K = range(2, 10)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, labels))

# Elbow & Silhouette plots
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method (PCA data)")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(K, sil_scores, marker='o')
plt.title("Silhouette Score vs k (PCA data)")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# Final KMeans with k=3
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(X_pca)

# Scatter plot in PCA space
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_pca, palette="Set2", alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (k=3) on PCA-reduced data")
plt.legend(title="Cluster")
plt.show()

# Evaluate clustering
sil_pca = silhouette_score(X_pca, labels_pca)
dbi_pca = davies_bouldin_score(X_pca, labels_pca)
print(f"Silhouette Score (k=3, PCA data): {sil_pca:.3f}")
print(f"Davies-Bouldin Index (k=3, PCA data): {dbi_pca:.3f}")
