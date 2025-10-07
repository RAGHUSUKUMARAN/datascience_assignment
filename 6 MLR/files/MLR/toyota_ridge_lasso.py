# toyota_ridge_lasso.py
# Apply RidgeCV and LassoCV to ToyotaCorolla dataset, evaluate and save coefficients/results.
# Save at: D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\toyota_ridge_lasso.py
# Requires: pandas, numpy, scikit-learn, statsmodels

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\ToyotaCorolla_MLR_cleaned.csv"
if not Path(DATA_PATH).exists():
    DATA_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\ToyotaCorolla - MLR.csv"

df = pd.read_csv(DATA_PATH)
if 'Price' not in df.columns:
    raise SystemExit("Target column 'Price' not found in CSV.")

# prepare X, y (drop helper columns if present)
drop_cols = ['Price_pos', 'log_Price', 'KM_pos']
X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ['Price'], errors='ignore')
y = pd.to_numeric(df['Price'], errors='coerce')
X = X.apply(pd.to_numeric, errors='coerce')

# align and drop NA rows
mask = X.dropna().index.intersection(y.dropna().index)
X = X.loc[mask].copy()
y = y.loc[mask].copy()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# scale numeric features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# common alpha grid
alphas = np.logspace(-4, 4, 100)

# RidgeCV (with built-in CV)
ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_s, y_train)
ridge_alpha = ridge_cv.alpha_
ridge_coef = ridge_cv.coef_
ridge_intercept = ridge_cv.intercept_
y_pred_ridge = ridge_cv.predict(X_test_s)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
ridge_r2 = r2_score(y_test, y_pred_ridge)

# LassoCV (with built-in CV)
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=10000, random_state=42).fit(X_train_s, y_train)
lasso_alpha = lasso_cv.alpha_
lasso_coef = lasso_cv.coef_
lasso_intercept = lasso_cv.intercept_
y_pred_lasso = lasso_cv.predict(X_test_s)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
lasso_r2 = r2_score(y_test, y_pred_lasso)

# results DataFrame
results = pd.DataFrame({
    'model': ['RidgeCV', 'LassoCV'],
    'alpha': [ridge_alpha, lasso_alpha],
    'MAE': [round(ridge_mae,3), round(lasso_mae,3)],
    'RMSE': [round(ridge_rmse,3), round(lasso_rmse,3)],
    'R2': [round(ridge_r2,4), round(lasso_r2,4)]
})

# coefficients table
coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'ridge_coef': np.round(ridge_coef, 6),
    'lasso_coef': np.round(lasso_coef, 6)
})

# save outputs
out_dir = Path(r"D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\ridge_lasso_results")
out_dir.mkdir(parents=True, exist_ok=True)
results.to_csv(out_dir / "ridge_lasso_metrics.csv", index=False)
coef_df.to_csv(out_dir / "ridge_lasso_coefficients.csv", index=False)

# print summary
print("Ridge alpha:", ridge_alpha)
print("Lasso alpha:", lasso_alpha)
print("\nEvaluation metrics:")
print(results.to_string(index=False))
print("\nTop coefficients (sorted by absolute Ridge coef):")
print(coef_df.assign(abs_ridge=lambda df: df.ridge_coef.abs()).sort_values('abs_ridge', ascending=False).head(20).to_string(index=False))

# Save trained models (optional - requires joblib)
try:
    import joblib
    joblib.dump(ridge_cv, out_dir / "ridge_cv_model.joblib")
    joblib.dump(lasso_cv, out_dir / "lasso_cv_model.joblib")
    print("\nSaved models to:", out_dir)
except Exception:
    print("\njoblib not available — models not saved. Install joblib to save models.")

# Quick note: to inspect non-zero lasso features
nonzero_lasso = coef_df[coef_df['lasso_coef'] != 0].sort_values('lasso_coef', key=lambda s: s.abs(), ascending=False)
print(f"\nLasso selected {len(nonzero_lasso)} non-zero features. Top ones:")
print(nonzero_lasso.head(10).to_string(index=False))
