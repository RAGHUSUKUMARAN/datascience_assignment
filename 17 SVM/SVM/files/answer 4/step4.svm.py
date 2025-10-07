"""
step4_svm.py
Task 4: SVM Implementation
- Loads data (either original CSV or pre-split X_train/X_test files if available)
- Encodes categorical variables (one-hot)
- Standardizes features
- Trains SVM with small hyperparameter search
- Evaluates and saves metrics, confusion matrix, and model
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ---------- User config ----------
CSV_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\mushroom.csv"
OUTPUT_DIR = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\correlation_outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2
USE_PRE_SPLIT = False  # If True, we'll load X_train/X_test CSVs if available
# ---------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    # If pre-split files exist and user chose to use them, load those
    if USE_PRE_SPLIT:
        p = Path(CSV_PATH).parent
        xtrain = p / "X_train.csv"
        xtest = p / "X_test.csv"
        ytrain = p / "y_train.csv"
        ytest = p / "y_test.csv"
        if xtrain.exists() and xtest.exists() and ytrain.exists() and ytest.exists():
            X_train = pd.read_csv(xtrain)
            X_test = pd.read_csv(xtest)
            y_train = pd.read_csv(ytrain).iloc[:, 0].values
            y_test = pd.read_csv(ytest).iloc[:, 0].values
            print("Loaded pre-split X_train/X_test/y_train/y_test from folder.")
            return X_train, X_test, y_train, y_test
        else:
            print("Pre-split files requested but not found; falling back to single CSV load.")

    # Load full CSV, encode, and split
    df = pd.read_csv(CSV_PATH)
    if 'class' not in df.columns:
        raise SystemExit("Target column 'class' not found in CSV.")

    y = df['class'].copy()
    X = df.drop(columns=['class'])

    # Label encode target (assuming 'e'/'p' or 'edible'/'poisonous')
    # Convert to 0/1
    y = y.astype(str)
    if set(y.unique()) <= set(['e', 'p']):
        y_encoded = (y == 'p').astype(int).values  # poisonous=1, edible=0
    else:
        # fallback: map unique values to 0/1 by sorted order
        unique = sorted(y.unique())
        mapping = {unique[0]: 0, unique[1]: 1}
        y_encoded = y.map(mapping).values
        print("Target mapping used:", mapping)

    # One-hot encode features (drop_first=False to preserve full info; scaler handles multicollinearity)
    X_encoded = pd.get_dummies(X, drop_first=False)
    print("Feature matrix after one-hot encoding shape:", X_encoded.shape)

    # train-test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Build pipeline: scaler + SVM
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=False))
    ])

    # Small grid for C and kernel (keeps run-time reasonable)
    param_grid = {
        "svc__C": [0.1, 1, 5],
        "svc__kernel": ["rbf", "linear"],
        "svc__gamma": ["scale"]  # keep default gamma
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    print("Starting GridSearchCV for SVM (this may take a bit)...")
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Best params:", grid.best_params_)
    # Predict
    y_pred = best.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cls_report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist()  # convert to list for easy saving
    }

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "svm_best_model.joblib")
    joblib.dump(grid.best_estimator_, model_path)
    print(f"Saved trained model to: {model_path}")

    # Save results
    results_df = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1_score"],
        "value": [acc, prec, rec, f1]
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, "svm_metrics_summary.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "svm_classification_report.txt"), "w") as f:
        f.write(cls_report)
    pd.DataFrame(cm, index=["actual_0","actual_1"], columns=["pred_0","pred_1"]).to_csv(
        os.path.join(OUTPUT_DIR, "svm_confusion_matrix.csv")
    )
    print("Saved metrics and confusion matrix to output folder.")

    # Print summary
    print("\n--- SVM Evaluation Summary ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:\n", cls_report)
    print("Confusion matrix:\n", cm)

    return results, grid.best_params_

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    results, best_params = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nALL DONE — outputs in:", OUTPUT_DIR)
