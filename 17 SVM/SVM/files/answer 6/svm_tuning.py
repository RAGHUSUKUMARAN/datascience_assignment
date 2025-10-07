# svm_tuning.py
"""
SVM hyperparameter tuning and evaluation script.

Assumptions:
- You have X (features) and y (labels). If not, uncomment the example using sklearn's iris dataset.
- Recommended: run inside a virtualenv with scikit-learn installed (>=0.24).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import time

# ---------- USER DATA LOADING ----------
# Option A: if you already have X,y (numpy arrays or pandas)
# from your_data_module import X, y

# Option B: quick example dataset (uncomment to test script immediately)
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# ---------- TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- COMMON PIPELINE ----------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=False))
])

# ---------- CROSS-VALIDATION SETUP ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- PARAM GRID FOR GRIDSEARCH ----------
# We separate grids by kernel to keep combinatorial explosion manageable.
param_grid = [
    {
        "svc__kernel": ["linear"],
        "svc__C": [0.01, 0.1, 1, 10, 100],
        "svc__class_weight": [None, "balanced"],
    },
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "svc__class_weight": [None, "balanced"],
    },
    {
        "svc__kernel": ["poly"],
        "svc__C": [0.1, 1, 10],
        "svc__degree": [2, 3, 4],
        "svc__gamma": ["scale", "auto"],
        "svc__coef0": [0.0, 0.1, 0.5],
        "svc__class_weight": [None],
    },
    {
        "svc__kernel": ["sigmoid"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", "auto", 0.01, 0.1],
        "svc__coef0": [0.0, 0.1, 0.5],
        "svc__class_weight": [None],
    }
]

# ---------- RANDOMIZED GRID (broader search) ----------
# If you have many features / dataset large, randomized search is faster.
param_dist = {
    "svc__kernel": ["rbf", "linear", "poly", "sigmoid"],
    "svc__C": [10**k for k in np.linspace(-3, 3, 11)],      # 1e-3 .. 1e3
    "svc__gamma": ["scale", "auto"]+ [10**k for k in np.linspace(-4, 0, 5)], # mix
    "svc__degree": [2, 3, 4],    # only relevant for poly
    "svc__coef0": [0.0, 0.1, 0.5],
    "svc__class_weight": [None, "balanced"],
}

# Note: We defined param_dist above but RandomizedSearchCV will ignore keys that
# don't apply to a kernel (e.g., degree for non-poly). That's okay.


# ---------- FUNCTIONS FOR TUNING & EVAL ----------
def run_grid_search(pipe, param_grid, X_train, y_train, cv, scoring="f1_macro", n_jobs=-1):
    print("Starting GridSearchCV...")
    t0 = time.time()
    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring=scoring,
        verbose=2, n_jobs=n_jobs, refit=True
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Grid search done in {elapsed:.1f} s")
    print("Best score (CV):", grid.best_score_)
    print("Best params:", grid.best_params_)
    return grid

def run_random_search(pipe, param_dist, X_train, y_train, cv, n_iter=40, scoring="f1_macro", n_jobs=-1):
    print("Starting RandomizedSearchCV...")
    t0 = time.time()
    rand = RandomizedSearchCV(
        pipe, param_dist, n_iter=n_iter, cv=cv, scoring=scoring,
        verbose=2, n_jobs=n_jobs, random_state=42, refit=True
    )
    rand.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Randomized search done in {elapsed:.1f} s")
    print("Best score (CV):", rand.best_score_)
    print("Best params:", rand.best_params_)
    return rand

def evaluate_model(model, X_test, y_test, target_names=None, save_cm_fig=True, cm_filename="confusion_matrix.png"):
    print("\n--- Test set evaluation ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix (test set)")
    if save_cm_fig:
        plt.savefig(cm_filename, bbox_inches="tight")
        print(f"Saved confusion matrix to {cm_filename}")
    plt.show()

# ---------- RUN TUNING (pick one or both) ----------
if __name__ == "__main__":
    # 1) Quick Randomized Search (broad)
    try:
        rand_search = run_random_search(
            pipe,
            param_dist,
            X_train, y_train,
            cv=cv,
            n_iter=30,      # reduce/increase based on compute budget
            scoring="f1_macro",
            n_jobs=-1
        )
    except Exception as e:
        print("Randomized search failed (likely due to param_dist construction).")
        print(e)
        rand_search = None

    # 2) More exhaustive Grid Search (fine-tuning)
    grid_search = run_grid_search(
        pipe,
        param_grid,
        X_train, y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    # Pick best model (prefer grid_search best if available)
    best_model = None
    if grid_search is not None:
        best_model = grid_search.best_estimator_
        print("\nSelected model from GridSearch.")
    elif rand_search is not None:
        best_model = rand_search.best_estimator_
        print("\nSelected model from RandomizedSearch.")
    else:
        raise RuntimeError("No tuned model available. Both searches failed.")

    # Evaluate on test set
    class_names = [str(c) for c in np.unique(y)]
    evaluate_model(best_model, X_test, y_test, target_names=class_names)

    # Save best model
    joblib.dump(best_model, "best_svm_model.joblib")
    print("Saved best model to best_svm_model.joblib")
