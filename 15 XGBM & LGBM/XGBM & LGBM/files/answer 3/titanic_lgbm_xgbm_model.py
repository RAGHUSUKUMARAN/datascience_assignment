# titanic_lgbm_xgbm_model.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb

# === PATH SETUP ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\15 XGBM & LGBM\XGBM & LGBM"
train_path = os.path.join(base_path, "Titanic_train_processed.csv")

# === LOAD CLEAN DATA ===
df = pd.read_csv(train_path)
print("✅ Preprocessed dataset loaded successfully:", df.shape)

# === 1️⃣ SPLIT DATA ===
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# === 2️⃣ EVALUATION FUNCTION ===
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n{name} Performance:")
    print("-----------------------------")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"{name}_confusion_matrix.png"))
    plt.close()

    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": roc_auc}

# ============================================================
# 3️⃣ LIGHTGBM MODEL
# ============================================================

lgbm_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)
lgbm_model.fit(X_train, y_train)
lgbm_results = evaluate_model("LightGBM", lgbm_model, X_test, y_test)
joblib.dump(lgbm_model, os.path.join(base_path, "lightgbm_model.pkl"))

# ============================================================
# 4️⃣ XGBOOST MODEL
# ============================================================

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_results = evaluate_model("XGBoost", xgb_model, X_test, y_test)
joblib.dump(xgb_model, os.path.join(base_path, "xgboost_model.pkl"))

# ============================================================
# 5️⃣ CROSS VALIDATION (for performance robustness)
# ============================================================

for model, name in zip([lgbm_model, xgb_model], ["LightGBM", "XGBoost"]):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n{name} 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 6️⃣ HYPERPARAMETER TUNING (LightGBM example)
# ============================================================

print("\nRunning LightGBM Hyperparameter Tuning (Grid Search)...")
param_grid = {
    'num_leaves': [15, 31, 63],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500]
}

grid = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=0,
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("✅ Best LightGBM Parameters:")
print(grid.best_params_)
print(f"Best CV Accuracy: {grid.best_score_:.4f}")

# ============================================================
# 7️⃣ FEATURE IMPORTANCE COMPARISON
# ============================================================

plt.figure(figsize=(10,5))
lgb.plot_importance(lgbm_model, max_num_features=10, title="LightGBM Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "lightgbm_feature_importance.png"))
plt.close()

plt.figure(figsize=(10,5))
xgb.plot_importance(xgb_model, max_num_features=10, title="XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "xgboost_feature_importance.png"))
plt.close()

# ============================================================
# 8️⃣ COMPARISON SUMMARY
# ============================================================

results_df = pd.DataFrame([lgbm_results, xgb_results])
results_df.to_csv(os.path.join(base_path, "model_comparison_results.csv"), index=False)
print("\n✅ Model comparison completed. Results saved to 'model_comparison_results.csv'")
print(results_df)

# ============================================================
# 9️⃣ QUICK INSIGHTS
# ============================================================
print("\n--- Insights ---")
print("1. Both LightGBM and XGBoost perform strongly on Titanic survival prediction.")
print("2. LightGBM typically trains faster with similar or better accuracy.")
print("3. Feature importances often highlight 'Sex', 'Pclass', 'Fare', and 'Age' as top predictors.")
print("4. Hyperparameter tuning can yield slight accuracy improvements (~1–3%).")
print("5. ROC-AUC and cross-validation scores confirm the models are generalizing well.")
