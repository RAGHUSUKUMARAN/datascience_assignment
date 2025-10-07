# train_decision_tree.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

# ----------------- CONFIG -----------------
BASE_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\13 decision tree\Decision Tree"
RAW_XLSX = os.path.join(BASE_PATH, "heart_disease.xlsx")
PROCESSED_CSV = os.path.join(BASE_PATH, "heart_processed.csv")

MODEL_OUT = os.path.join(BASE_PATH, "decision_tree_baseline.pkl")
REPORT_OUT = os.path.join(BASE_PATH, "decision_tree_evaluation.txt")
CM_PNG = os.path.join(BASE_PATH, "confusion_matrix_baseline.png")
ROC_PNG = os.path.join(BASE_PATH, "roc_curve_baseline.png")
TREE_PNG = os.path.join(BASE_PATH, "decision_tree_baseline.png")

RANDOM_STATE = 42
TEST_SIZE = 0.2
# ------------------------------------------

def ensure_processed():
    """If processed CSV missing, create it from raw Excel (simple FE)."""
    if os.path.exists(PROCESSED_CSV):
        print(f"Found processed file: {PROCESSED_CSV}")
        return pd.read_csv(PROCESSED_CSV)
    print("Processed CSV not found — creating from raw Excel (light feature engineering)...")
    df = pd.read_excel(RAW_XLSX, sheet_name="Heart_disease")
    # Basic fixes similar to previous step
    df['oldpeak'] = df['oldpeak'].fillna(df['oldpeak'].median())
    for col in ['trestbps', 'chol']:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())
    df['sex'] = df['sex'].map({'Male':1, 'Female':0})
    for col in ['fbs','exang']:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].map({'True':1,'TURE':1,'False':0,'FALSE':0}).fillna(0).astype(int)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    cat_cols = [c for c in ['cp','restecg','slope','thal'] if c in df.columns]
    df_processed = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    if 'num' in df_processed.columns:
        df_processed = df_processed.drop(columns=['num'])
    df_processed.to_csv(PROCESSED_CSV, index=False)
    print("Saved processed CSV to:", PROCESSED_CSV)
    return df_processed

def train_and_evaluate(df):
    # prepare X and y
    if 'target' not in df.columns:
        raise ValueError("No 'target' column found in processed data.")
    X = df.drop(columns=['target'])
    y = df['target']

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("Train/test sizes:", X_train.shape, X_test.shape)

    # baseline Decision Tree
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # predictions & probs
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # save model
    joblib.dump(clf, MODEL_OUT)

    # write evaluation report
    with open(REPORT_OUT, "w") as f:
        f.write("Decision Tree — Baseline Evaluation\n")
        f.write(f"Train shape: {X_train.shape}\nTest shape: {X_test.shape}\n\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))
    print("Saved model to:", MODEL_OUT)
    print("Saved evaluation report to:", REPORT_OUT)

    # plot confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Disease", "Disease"], rotation=45)
    plt.yticks(tick_marks, ["No Disease", "Disease"])
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(CM_PNG)
    plt.close()
    print("Saved confusion matrix to:", CM_PNG)

    # ROC curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc_val:.3f})')
        plt.plot([0,1], [0,1], linestyle='--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_PNG)
        plt.close()
        print("Saved ROC curve to:", ROC_PNG)
    else:
        print("No probability estimates available; skipped ROC curve.")

    # Save a visual of the decision tree (may be large)
    try:
        plt.figure(figsize=(18,10))
        plot_tree(clf, feature_names=X.columns, class_names=["No","Yes"], filled=True, rounded=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(TREE_PNG)
        plt.close()
        print("Saved decision tree visualization to:", TREE_PNG)
    except Exception as e:
        print("Could not save tree visualization:", e)

    # Print summary to console
    print("\n=== Metrics summary ===")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    df_proc = ensure_processed()
    train_and_evaluate(df_proc)
