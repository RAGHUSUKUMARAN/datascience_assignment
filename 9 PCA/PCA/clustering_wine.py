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

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different K values
inertia = []
silhouette_scores = []
dbi_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    dbi_scores.append(davies_bouldin_score(X_scaled, labels))

# Plot Elbow & Silhouette
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method (Inertia)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(K, silhouette_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# Choose k=3 (based on evaluation)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="Set2", alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (k=3) visualized on PCA-reduced space")
plt.legend(title="Cluster")
plt.show()

# Evaluate clustering
sil_score = silhouette_score(X_scaled, labels)
dbi_score = davies_bouldin_score(X_scaled, labels)
print(f"Silhouette Score (k=3): {sil_score:.3f}")
print(f"Davies-Bouldin Index (k=3): {dbi_score:.3f}")
