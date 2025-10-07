# step5_visualize_results.py
"""
Visualize SVM classification results.
Saves plots to OUTPUT_DIR.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------- User config ----------
OUTPUT_DIR = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\correlation_outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "svm_best_model.joblib")
CSV_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\mushroom.csv"  # fallback if pre-split not used
USE_PRE_SPLIT = False  # if you saved X_test/y_test CSVs set True and script will try to load them
PLOT_DPI = 150
RANDOM_STATE = 42
# ---------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

# ---------- Load model and data ----------
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found at {MODEL_PATH}. Run Task 4 to save svm_best_model.joblib first.")

model = joblib.load(MODEL_PATH)
print("Loaded model:", model)

# Load test data (prefer pre-split files if available)
if USE_PRE_SPLIT:
    base = os.path.dirname(CSV_PATH)
    x_test_path = os.path.join(base, "X_test.csv")
    y_test_path = os.path.join(base, "y_test.csv")
    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).iloc[:, 0].values
    else:
        raise SystemExit("Pre-split test files requested but not found.")
else:
    # load full CSV and split here to reproduce same split as Task 4
    df = pd.read_csv(CSV_PATH)
    if 'class' not in df.columns:
        raise SystemExit("Target column 'class' not found in CSV.")
    y = df['class'].astype(str)
    X = df.drop(columns=['class'])
    # encode y to 0/1 same logic as training script
    if set(y.unique()) <= set(['e', 'p']):
        y_encoded = (y == 'p').astype(int).values
    else:
        unique = sorted(y.unique())
        mapping = {unique[0]: 0, unique[1]: 1}
        y_encoded = y.map(mapping).values
    X_encoded = pd.get_dummies(X, drop_first=False)
    # ensure columns line up (if training used same encoding)
    # If model was trained on a different feature set, prefer using saved pre-split CSVs.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

# Convert to numpy arrays
X_test_arr = np.asarray(X_test)
y_test_arr = np.asarray(y_test)

# ---------- Predictions ----------
y_pred = model.predict(X_test_arr)

# Confusion matrix
cm = confusion_matrix(y_test_arr, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Save confusion matrix (counts)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["pred_0","pred_1"], yticklabels=["true_0","true_1"])
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix (counts)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "svm_confusion_matrix_counts.png"), dpi=PLOT_DPI)
plt.close()

# Save normalized confusion matrix (percent)
plt.figure(figsize=(5,4))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False,
            xticklabels=["pred_0","pred_1"], yticklabels=["true_0","true_1"])
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix (normalized)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "svm_confusion_matrix_normalized.png"), dpi=PLOT_DPI)
plt.close()

# Classification report heatmap (turn report into a dataframe)
report = classification_report(y_test_arr, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
# Save textual report
with open(os.path.join(OUTPUT_DIR, "svm_classification_report.txt"), "w") as f:
    f.write(classification_report(y_test_arr, y_pred, zero_division=0))
report_df.to_csv(os.path.join(OUTPUT_DIR, "svm_classification_report_table.csv"))

# Plot classification report (precision, recall, f1) for classes
metrics_df = report_df.loc[['0','1'], ['precision','recall','f1-score']].astype(float)
plt.figure(figsize=(6,4))
metrics_df.plot(kind='bar')
plt.title("Precision / Recall / F1-score by Class")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "svm_class_metrics_bar.png"), dpi=PLOT_DPI)
plt.close()

# ---------- ROC curve & AUC ----------
y_score = None
if hasattr(model, "decision_function"):
    try:
        y_score = model.decision_function(X_test_arr)
    except Exception:
        y_score = None

if y_score is None and hasattr(model, "predict_proba"):
    try:
        y_score = model.predict_proba(X_test_arr)[:, 1]
    except Exception:
        y_score = None

if y_score is not None:
    fpr, tpr, _ = roc_curve(y_test_arr, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1], [0,1], linestyle='--', color='gray', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "svm_roc_auc.png"), dpi=PLOT_DPI)
    plt.close()
    print("Saved ROC curve. AUC:", roc_auc)
else:
    print("Model does not expose decision_function or predict_proba. ROC curve skipped.")

# ---------- 2D Embeddings (PCA and t-SNE) colored by true/predicted ----------
# PCA (fast)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
try:
    X_2d_pca = pca.fit_transform(X_test_arr)
    df_plot = pd.DataFrame({
        'pc1': X_2d_pca[:,0],
        'pc2': X_2d_pca[:,1],
        'true': y_test_arr,
        'pred': y_pred
    })
    # True labels
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df_plot, x='pc1', y='pc2', hue='true', style='true', s=40, palette='deep')
    plt.title('PCA 2D - True labels')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "svm_pca_true_labels.png"), dpi=PLOT_DPI)
    plt.close()

    # Predicted labels
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df_plot, x='pc1', y='pc2', hue='pred', style='pred', s=40, palette='deep')
    plt.title('PCA 2D - Predicted labels')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "svm_pca_pred_labels.png"), dpi=PLOT_DPI)
    plt.close()
except Exception as e:
    print("PCA embedding failed:", e)

# t-SNE (slower; sample if too big)
tsne_n = min(2000, X_test_arr.shape[0])  # cap samples for speed
if X_test_arr.shape[0] > tsne_n:
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(X_test_arr.shape[0], size=tsne_n, replace=False)
    X_sample = X_test_arr[sample_idx]
    y_sample = y_test_arr[sample_idx]
    y_pred_sample = y_pred[sample_idx]
else:
    X_sample = X_test_arr
    y_sample = y_test_arr
    y_pred_sample = y_pred

try:
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, init='pca')
    X_2d_tsne = tsne.fit_transform(X_sample)
    df_tsne = pd.DataFrame({
        'tsne1': X_2d_tsne[:,0],
        'tsne2': X_2d_tsne[:,1],
        'true': y_sample,
        'pred': y_pred_sample
    })
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df_tsne, x='tsne1', y='tsne2', hue='true', s=30, palette='tab10')
    plt.title('t-SNE (true labels)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "svm_tsne_true_labels.png"), dpi=PLOT_DPI)
    plt.close()

    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df_tsne, x='tsne1', y='tsne2', hue='pred', s=30, palette='tab10')
    plt.title('t-SNE (predicted labels)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "svm_tsne_pred_labels.png"), dpi=PLOT_DPI)
    plt.close()
except Exception as e:
    print("t-SNE embedding failed or too slow:", e)

print("All visualizations saved to:", OUTPUT_DIR)
