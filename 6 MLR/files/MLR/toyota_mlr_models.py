# toyota_mlr_models.py
# Build 3+ regression models (OLS, OLS with backward selection, Ridge) and evaluate them.
# Expects either saved splits in D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\splits
# or will load the cleaned CSV and create a fresh train/test split.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- CONFIG ----------
SPLITS_DIR = Path(r"D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\splits")
CLEANED_CSV = Path(r"D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\ToyotaCorolla_MLR_cleaned.csv")
ORIG_CSV = Path(r"D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\ToyotaCorolla - MLR.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------- Helpers ----------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def print_metrics(name, y_true, y_pred):
    print(f"\n{name} performance:")
    print(" MAE:  ", round(mean_absolute_error(y_true, y_pred), 3))
    print(" RMSE: ", round(rmse(y_true, y_pred), 3))
    print(" R2:   ", round(r2_score(y_true, y_pred), 4))

def fit_ols_and_report(X_train, y_train, X_test, y_test, model_name="OLS"):
    Xtr_sm = add_constant(X_train, has_constant='add')
    ols = sm.OLS(y_train.astype(float), Xtr_sm.astype(float)).fit()
    print(f"\n=== {model_name} summary ===")
    print(ols.summary())
    preds = ols.predict(add_constant(X_test, has_constant='add').astype(float))
    print_metrics(model_name, y_test, preds)
    coef_table = pd.DataFrame({
        'feature': ['const'] + list(X_train.columns),
        'coef': np.round(ols.params.values, 4)
    })
    print("\nCoefficients:")
    print(coef_table.to_string(index=False))
    return ols, preds

# ---------- Load data / splits ----------
if SPLITS_DIR.exists() and (SPLITS_DIR / "X_train.csv").exists():
    print("Loading saved splits from:", SPLITS_DIR)
    X_train = pd.read_csv(SPLITS_DIR / "X_train.csv")
    X_test  = pd.read_csv(SPLITS_DIR / "X_test.csv")
    y_train = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze()
else:
    # load cleaned if exists else original
    if CLEANED_CSV.exists():
        df = pd.read_csv(CLEANED_CSV)
    else:
        df = pd.read_csv(ORIG_CSV)
        # minimal cleaning: ensure numeric columns and dummies for Fuel_Type if present
        for c in df.columns:
            if c != "Fuel_Type":
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if "Fuel_Type" in df.columns:
            df["Fuel_Type"] = df["Fuel_Type"].astype(str).str.strip()
            df = pd.get_dummies(df, columns=["Fuel_Type"], drop_first=True)
        df = df.dropna(subset=["Price"])
    # define features: use all numeric except Price (the cleaned file includes prepared columns)
    target = "Price"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(df[target], errors='coerce')
    mask = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[mask]
    y = y.loc[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Created fresh split: Train {X_train.shape}, Test {X_test.shape}")

# Make sure indices are aligned and types numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test  = X_test.apply(pd.to_numeric, errors='coerce')
y_train = pd.to_numeric(y_train, errors='coerce')
y_test  = pd.to_numeric(y_test, errors='coerce')

# Drop rows with NaNs just in case
train_idx = X_train.dropna().index.intersection(y_train.dropna().index)
test_idx  = X_test.dropna().index.intersection(y_test.dropna().index)
X_train = X_train.loc[train_idx].copy(); y_train = y_train.loc[train_idx].copy()
X_test  = X_test.loc[test_idx].copy();  y_test  = y_test.loc[test_idx].copy()

print("\nFinal feature set used:", list(X_train.columns))
print("Train size:", X_train.shape, " Test size:", X_test.shape)

# ---------- MODEL 1: Baseline OLS (all features) ----------
ols_all, preds_all = fit_ols_and_report(X_train, y_train, X_test, y_test, model_name="Model A - OLS (all features)")

# ---------- MODEL 2: Backward elimination (p-value) OLS ----------
# iterative removal of highest p-value > threshold (0.05)
print("\n--- Building Model B via backward elimination (p-value) ---")
Xb = X_train.copy()
yb = y_train.copy()
p_thresh = 0.05
while True:
    Xb_sm = add_constant(Xb, has_constant='add')
    model = sm.OLS(yb.astype(float), Xb_sm.astype(float)).fit()
    pvals = model.pvalues.drop('const', errors='ignore')
    if pvals.empty:
        break
    max_p = pvals.max()
    if max_p > p_thresh:
        drop_col = pvals.idxmax()
        print(f" Dropping {drop_col} with p-value {max_p:.4f}")
        Xb = Xb.drop(columns=[drop_col])
    else:
        break

ols_sel, preds_sel = fit_ols_and_report(Xb, yb, X_test[Xb.columns], y_test, model_name="Model B - OLS (backward selection)")

# ---------- MODEL 3: Ridge regression (regularized) ----------
print("\n--- Building Model C: RidgeCV (with scaling) ---")
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xte_s = scaler.transform(X_test)

alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(Xtr_s, y_train)
print("Ridge chosen alpha:", ridge_cv.alpha_)
preds_ridge = ridge_cv.predict(Xte_s)
print_metrics = print_metrics  # alias
print_metrics("Model C - RidgeCV", y_test, preds_ridge)

# coefficients for ridge (map back to feature names)
ridge_coefs = pd.DataFrame({
    'feature': list(X_train.columns),
    'ridge_coef': np.round(ridge_cv.coef_, 4)
})
print("\nRidge coefficients:")
print(ridge_coefs.to_string(index=False))

# ---------- Compare models on test set ----------
print("\n=== Summary comparison on test set ===")
print_metrics("Model A - OLS (all features)", y_test, preds_all)
print_metrics("Model B - OLS (selected)", y_test, preds_sel)
print_metrics("Model C - RidgeCV", y_test, preds_ridge)

# Save model coefficient tables for reporting
coef_A = pd.DataFrame({'feature': ['const'] + list(X_train.columns), 'coef_A': np.round(ols_all.params.values,4)})
coef_B = pd.DataFrame({'feature': ['const'] + list(Xb.columns), 'coef_B': np.round(ols_sel.params.values,4)})
coef_R = ridge_coefs

coef_out = Path(r"D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR").joinpath("model_coefficients_summary.csv")
coef_df = coef_A.merge(coef_B, on='feature', how='outer').merge(coef_R, on='feature', how='outer')
coef_df.to_csv(coef_out, index=False)
print("\nSaved coefficient summary to:", coef_out)
