# model_evaluation_heart.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.tree import plot_tree

# === PATHS ===
BASE_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\13 decision tree\Decision Tree"
PROCESSED_CSV = os.path.join(BASE_PATH, "heart_processed.csv")
MODEL_PATH = os.path.join(BASE_PATH, "decision_tree_best.pkl")

# === OUTPUT FILES ===
EVAL_TXT = os.path.join(BASE_PATH, "decision_tree_final_evaluation.txt")
CM_PNG = os.path.join(BASE_PATH, "confusion_matrix_final.png")
ROC_PNG = os.path.join(BASE_PATH, "roc_curve_final.png")
FI_PNG = os.path.join(BASE_PATH, "feature_importances_final.png")
TREE_PNG = os.path.join(BASE_PATH, "decision_tree_final.png")

# === LOAD MODEL & DATA ===
print("Loading model and data...")
df = pd.read_csv(PROCESSED_CSV)
model = joblib.load(MODEL_PATH)

# === PREPARE FEATURES ===
X = df.drop(columns=['target'])
y = df['target']

# === MAKE PREDICTIONS ===
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

# === METRICS ===
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, y_proba) if y_proba is not None else float("nan")

cm = confusion_matrix(y, y_pred)

# === SAVE METRICS REPORT ===
with open(EVAL_TXT, "w") as f:
    f.write("=== Decision Tree Final Evaluation ===\n\n")
    f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y, y_pred, zero_division=0))
print("Saved evaluation metrics to:", EVAL_TXT)

# === CONFUSION MATRIX PLOT ===
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix — Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(CM_PNG)
plt.close()
print("Saved confusion matrix plot:", CM_PNG)

# === ROC CURVE ===
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc_val:.3f})")
    plt.plot([0,1], [0,1], linestyle='--', lw=1, color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Decision Tree")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PNG)
    plt.close()
    print("Saved ROC curve:", ROC_PNG)

# === FEATURE IMPORTANCES ===
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=importances.head(15), y=importances.head(15).index)
plt.title("Top 15 Feature Importances — Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(FI_PNG)
plt.close()
print("Saved feature importance chart:", FI_PNG)

# === DECISION TREE STRUCTURE ===
plt.figure(figsize=(22,12))
plot_tree(model, feature_names=X.columns, class_names=["No Disease", "Disease"],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Structure — Final Model")
plt.tight_layout()
plt.savefig(TREE_PNG)
plt.close()
print("Saved decision tree visualization:", TREE_PNG)

# === SUMMARY IN CONSOLE ===
print("\n=== Model Performance Summary ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nFeature Importances (Top 10):")
print(importances.head(10))
print("\nAll evaluation files are saved in:", BASE_PATH)
