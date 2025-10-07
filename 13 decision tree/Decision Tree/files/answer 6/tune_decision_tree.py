# tune_decision_tree.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ---------------- CONFIG ----------------
BASE_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\13 decision tree\Decision Tree"
RAW_XLSX = os.path.join(BASE_PATH, "heart_disease.xlsx")
PROCESSED_CSV = os.path.join(BASE_PATH, "heart_processed.csv")

OUT_BEST_MODEL = os.path.join(BASE_PATH, "decision_tree_best.pkl")
OUT_CV_RESULTS = os.path.join(BASE_PATH, "gridsearch_cv_results.csv")
OUT_BEST_PARAMS = os.path.join(BASE_PATH, "best_params.txt")
OUT_REPORT = os.path.join(BASE_PATH, "decision_tree_tuned_report.txt")
OUT_FI_PNG = os.path.join(BASE_PATH, "feature_importances.png")
OUT_TREE_PNG = os.path.join(BASE_PATH, "decision_tree_best.png")
RANDOM_STATE = 42
TEST_SIZE = 0.2
# ----------------------------------------

def ensure_processed():
    """Create processed CSV if missing (light FE similar to earlier steps)."""
    if os.path.exists(PROCESSED_CSV):
        print("Found processed CSV:", PROCESSED_CSV)
        return pd.read_csv(PROCESSED_CSV)
    print("Processed CSV not found. Generating from raw Excel...")
    df = pd.read_excel(RAW_XLSX, sheet_name="Heart_disease")
    df['oldpeak'] = df['oldpeak'].fillna(df['oldpeak'].median())
    for col in ['trestbps', 'chol']:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())
    df['sex'] = df['sex'].map({'Male':1, 'Female':0})
    for col in ['fbs','exang']:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].map({'True':1,'TURE':1,'False':0,'FALSE':0}).fillna(0).astype(int)
    df['target'] = df['num'].apply(lambda x: 1 if x>0 else 0)
    cat_cols = [c for c in ['cp','restecg','slope','thal'] if c in df.columns]
    df_processed = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    if 'num' in df_processed.columns:
        df_processed = df_processed.drop(columns=['num'])
    df_processed.to_csv(PROCESSED_CSV, index=False)
    print("Saved processed CSV to:", PROCESSED_CSV)
    return df_processed

def main():
    df = ensure_processed()
    if 'target' not in df.columns:
        raise RuntimeError("Processed data missing 'target' column.")
    X = df.drop(columns=['target'])
    y = df['target']

    # train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print("Train/test shapes:", X_train.shape, X_test.shape)

    # parameter grid (sane but not huge)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 7, 9, 12],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': [None, 'sqrt', 'log2']
    }

    # Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    base_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("Starting GridSearchCV... (this may take a few minutes)")
    grid.fit(X_train, y_train)
    print("GridSearchCV done.")

    # Save CV results
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(OUT_CV_RESULTS, index=False)
    print("Saved CV results to:", OUT_CV_RESULTS)

    # Best estimator & params
    best = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    with open(OUT_BEST_PARAMS, "w") as f:
        f.write(f"Best ROC-AUC (CV): {best_score:.5f}\n")
        f.write("Best params:\n")
        for k,v in best_params.items():
            f.write(f"{k}: {v}\n")
    print("Saved best params to:", OUT_BEST_PARAMS)

    # Save best model
    joblib.dump(best, OUT_BEST_MODEL)
    print("Saved best model to:", OUT_BEST_MODEL)

    # Evaluate on test set
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:,1] if hasattr(best, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    cm = confusion_matrix(y_test, y_pred)

    # Save evaluation report
    with open(OUT_REPORT, "w") as f:
        f.write("Decision Tree — Tuned Model Evaluation\n\n")
        f.write(f"Test shape: {X_test.shape}\n\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))
    print("Saved evaluation report to:", OUT_REPORT)

    # Feature importances plot
    fi = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    fi.head(20).plot(kind='bar')
    plt.title("Top 20 Feature Importances (Decision Tree - tuned)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_FI_PNG)
    plt.close()
    print("Saved feature importances to:", OUT_FI_PNG)

    # Save tree visualization (may be large)
    try:
        plt.figure(figsize=(20,12))
        plot_tree(best, feature_names=X.columns, class_names=["No Disease","Disease"],
                  filled=True, rounded=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_TREE_PNG)
        plt.close()
        print("Saved tree visualization to:", OUT_TREE_PNG)
    except Exception as e:
        print("Could not save tree visualization:", e)

    # Quick console summary
    print("\n=== Test set performance (tuned model) ===")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  ROC-AUC: {roc_auc:.4f}")
    print("Done. All outputs in:", BASE_PATH)

if __name__ == "__main__":
    main()
