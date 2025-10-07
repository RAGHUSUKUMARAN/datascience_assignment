

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional nicer plots if seaborn is available
try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------- CONFIG --------------
# Change this to your actual path if needed.
# file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\8 clustering\Clustering\EastWestAirlines.csv"
# If you prefer CSV, comment above and uncomment below:
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\8 clustering\Clustering\EastWestAirlines.csv"

out_folder = Path(file_path).parent  # save outputs next to the data file
os.makedirs(out_folder, exist_ok=True)

# -------------- Load data --------------
def load_data(fp):
    fp = Path(fp)
    if fp.suffix.lower() in [".xlsx", ".xls"]:
        # try to be explicit about engine to avoid pandas warnings
        try:
            df = pd.read_excel(fp, sheet_name="data", engine="openpyxl")
        except Exception as e:
            # fallback: try without engine (pandas will try its default)
            print("Warning: read_excel with engine failed:", e)
            df = pd.read_excel(fp, sheet_name="data")
    elif fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp)
    else:
        raise ValueError("Unsupported file type: " + str(fp.suffix))
    return df

print("Loading data from:", file_path)
df = load_data(file_path)
print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# If ID# exists, drop it (identifier)
if 'ID#' in df.columns:
    df = df.drop(columns=['ID#'])
    print("Dropped ID# column. New shape:", df.shape)

# -------------- Quick checks --------------
print("\nMissing values per column:\n", df.isna().sum())

# Keep only numeric features for clustering/EDA visuals (but keep Award? separately)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns detected:", numeric_cols)

if len(numeric_cols) < 2:
    raise SystemExit("Not enough numeric columns found for EDA/Clustering. Please check the data sheet.")

df_num = df[numeric_cols].copy()

# -------------- Descriptive statistics --------------
desc = df_num.describe().T
desc_file = out_folder / "eastwest_descriptive_stats.csv"
desc.to_csv(desc_file)
print(f"\nDescriptive statistics saved to: {desc_file}")
print(desc)

# -------------- Histograms (before scaling) --------------
plt.figure(figsize=(12, 8))
df_num.hist(bins=30, figsize=(12, 8))
plt.suptitle("Histograms of numeric features (raw)", y=0.95)
plt.tight_layout()
hist_file = out_folder / "histograms_raw.png"
plt.savefig(hist_file, dpi=150)
plt.close()
print("Saved histograms to:", hist_file)

# -------------- Boxplots (detect outliers visually) --------------
plt.figure(figsize=(12, 6))
if HAS_SEABORN:
    sns.boxplot(data=df_num, orient="h")
else:
    df_num.plot(kind="box", vert=False, figsize=(12,6))
plt.title("Boxplots of numeric features (raw)")
plt.tight_layout()
box_file = out_folder / "boxplots_raw.png"
plt.savefig(box_file, dpi=150)
plt.close()
print("Saved boxplots to:", box_file)

# -------------- Correlation heatmap --------------
corr = df_num.corr()
corr_file_csv = out_folder / "eastwest_correlation.csv"
corr.to_csv(corr_file_csv)
plt.figure(figsize=(10, 8))
if HAS_SEABORN:
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", square=True)
else:
    plt.imshow(corr, cmap="coolwarm", interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation matrix")
plt.tight_layout()
corr_img = out_folder / "correlation_heatmap.png"
plt.savefig(corr_img, dpi=150)
plt.close()
print("Saved correlation heatmap to:", corr_img)
print("Correlation CSV saved to:", corr_file_csv)

# -------------- Pairwise scatter (sampled if many rows) --------------
max_pairs = 6  # limit number of vars for pairplot/pairs to keep plots readable
cols_for_pairs = numeric_cols if len(numeric_cols) <= max_pairs else numeric_cols[:max_pairs]
sample = df_num[cols_for_pairs].sample(n=min(1000, df_num.shape[0]), random_state=42)  # sampling for speed

if HAS_SEABORN:
    sns.pairplot(sample, diag_kind="hist", plot_kws=dict(s=20, alpha=0.6))
    pair_file = out_folder / "pairplot_sample.png"
    plt.gcf().set_size_inches(12, 10)
    plt.savefig(pair_file, dpi=150)
    plt.close()
else:
    pd.plotting.scatter_matrix(sample, alpha=0.5, figsize=(12, 12), diagonal='hist')
    pair_file = out_folder / "scatter_matrix_sample.png"
    plt.savefig(pair_file, dpi=150)
    plt.close()
print("Saved pairwise/sample scatter to:", pair_file)

# -------------- PCA for 2D visualization (with scaling) --------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num.fillna(df_num.median()))

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("\nPCA explained variance ratios (first 2):", pca.explained_variance_ratio_)

# Scatter plot of PCA (color by Award? if present)
plt.figure(figsize=(8,6))
if 'Award?' in df.columns:
    # try to use Award? as categorical coloring if present
    labels = df['Award?'].astype(str).values
    # map labels to integers for color mapping
    unique_labels = np.unique(labels)
    label_map = {lab:i for i,lab in enumerate(unique_labels)}
    colors = [label_map[l] for l in labels]
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=colors, alpha=0.6, s=20)
    # legend
    handles = []
    for lab, i in label_map.items():
        handles.append(plt.Line2D([0],[0], marker='o', color='w', label=str(lab),
                                  markerfacecolor=plt.cm.tab10(i % 10), markersize=6))
    plt.legend(handles=handles, title="Award?")
else:
    plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6, s=20)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (2D) visualization of numeric features")
pca_file = out_folder / "pca_2d.png"
plt.tight_layout()
plt.savefig(pca_file, dpi=150)
plt.close()
print("Saved PCA 2D plot to:", pca_file)

# -------------- Save scaled numeric data for clustering stage --------------
scaled_df = pd.DataFrame(X_scaled, columns=df_num.columns, index=df_num.index)
scaled_out = out_folder / "eastwest_scaled_numeric.csv"
scaled_df.to_csv(scaled_out, index=False)
print("Saved scaled numeric features to:", scaled_out)

# -------------- Summary print for report --------------
print("\n--- EDA SUMMARY ---")
print("Rows:", df.shape[0], "Numeric cols:", len(numeric_cols))
print("Saved outputs in:", out_folder)
print("Files created:")
for f in [hist_file, box_file, corr_img, pair_file, pca_file, scaled_out, desc_file, corr_file_csv]:
    print(" -", f)
print("\nYou can include the above PNGs and CSVs in your report. Next, we can run clustering on the scaled features (eastwest_scaled_numeric.csv).")
