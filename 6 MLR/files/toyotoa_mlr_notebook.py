
# toyota_mlr_final_corrected.py
# Corrected Multiple Linear Regression script for Toyota Corolla dataset
# CSV expected at: D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\ToyotaCorolla - MLR.csv
# Requires: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\6 MLR\MLR\ToyotaCorolla - MLR.csv"

df = pd.read_csv(DATA_PATH)

# Normalize/rename common column variants
for col in list(df.columns):
    low = col.strip().lower()
    if low in ('fuel type','fuel_type','fuel-type'):
        df.rename(columns={col: 'Fuel_Type'}, inplace=True)
    if low == 'age' and 'Age_08_04' not in df.columns:
        df.rename(columns={col: 'Age_08_04'}, inplace=True)

# Coerce numeric-like columns (except Fuel_Type) to numeric
for c in df.columns:
    if c != 'Fuel_Type':
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Normalize Fuel_Type strings
if 'Fuel_Type' in df.columns:
    df['Fuel_Type'] = df['Fuel_Type'].astype(str).str.strip().replace({'nan': np.nan})

if 'Price' not in df.columns:
    raise ValueError("Target column 'Price' not found.")
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])

# Impute numeric missing with median
for c in df.select_dtypes(include=[np.number]).columns:
    if df[c].isna().sum() > 0:
        df[c].fillna(df[c].median(), inplace=True)

if 'Fuel_Type' in df.columns and df['Fuel_Type'].isna().sum() > 0:
    df['Fuel_Type'].fillna(df['Fuel_Type'].mode().iloc[0], inplace=True)

candidate_features = ['Age_08_04', 'KM', 'HP', 'Automatic', 'cc', 'Doors', 'Weight', 'Quarterly_Tax', 'Fuel_Type']
FEATURES = [c for c in candidate_features if c in df.columns]
TARGET = 'Price'

X = df[FEATURES].copy()
y = df[TARGET].copy()

if 'Automatic' in X.columns:
    X['Automatic'] = X['Automatic'].replace({'Yes': 1, 'No': 0})
    X['Automatic'] = pd.to_numeric(X['Automatic'], errors='coerce').fillna(0).astype(int)

if 'Fuel_Type' in X.columns:
    X['Fuel_Type'] = X['Fuel_Type'].astype(str)
    X = pd.get_dummies(X, columns=['Fuel_Type'], drop_first=True)

X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

good_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[good_idx].copy()
y = y.loc[good_idx].copy()

# Drop zero-variance columns
zero_var_cols = X.columns[X.std(axis=0, ddof=0) == 0].tolist()
if zero_var_cols:
    X.drop(columns=zero_var_cols, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

def print_eval(y_true, y_pred, label=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")
    return mae, rmse, r2

def fit_sm_ols(Xt, yt, print_summary=True):
    Xt = Xt.astype(float)
    yt = yt.astype(float)
    Xt_sm = sm.add_constant(Xt, has_constant='add')
    model = sm.OLS(yt, Xt_sm).fit()
    if print_summary:
        print(model.summary())
    return model

# Model A: Baseline OLS
modelA = fit_sm_ols(X_train, y_train)
y_pred_A = modelA.predict(sm.add_constant(X_test))
print_eval(y_test, y_pred_A, label="Model A (OLS)")

# VIF (robust)
X_vif = X_train.copy().astype(float)
small_var_cols = [c for c in X_vif.columns if X_vif[c].std() < 1e-8]
if small_var_cols:
    X_vif = X_vif.drop(columns=small_var_cols)
vif_records = []
for i, col in enumerate(X_vif.columns):
    try:
        vif_val = variance_inflation_factor(X_vif.values, i)
        vif_records.append((col, float(vif_val)))
    except Exception:
        try:
            exog = X_vif.drop(columns=[col])
            endog = X_vif[col]
            exog_sm = sm.add_constant(exog, has_constant='add')
            r2 = sm.OLS(endog, exog_sm).fit().rsquared
            vif_val = 1.0 / (1.0 - r2) if (1.0 - r2) > 1e-12 else np.inf
            vif_records.append((col, float(vif_val)))
        except Exception:
            vif_records.append((col, np.nan))
vif_df = pd.DataFrame(vif_records, columns=['feature', 'VIF']).sort_values('VIF', ascending=False)
print("\nVIF:\n", vif_df.to_string(index=False))

vif_threshold = 10.0
low_vif_features = vif_df[vif_df['VIF'] < vif_threshold]['feature'].tolist()
if not low_vif_features:
    low_vif_features = X_train.columns.tolist()

# Model B: OLS on low-VIF features
modelB = fit_sm_ols(X_train[low_vif_features], y_train)
y_pred_B = modelB.predict(sm.add_constant(X_test[low_vif_features]))
print_eval(y_test, y_pred_B, label="Model B (OLS low-VIF)")

# Model C: log(target) with scaled KM and KM^2 to avoid numerical issues
X_train_c = X_train.copy()
X_test_c = X_test.copy()
if 'KM' in X_train_c.columns:
    X_train_c['KM_k'] = X_train_c['KM'] / 1000.0
    X_test_c['KM_k'] = X_test_c['KM'] / 1000.0
    X_train_c['KM_k_sq'] = X_train_c['KM_k'] ** 2
    X_test_c['KM_k_sq'] = X_test_c['KM_k'] ** 2

y_train_log = np.log(y_train.clip(lower=1))
modelC = fit_sm_ols(X_train_c, y_train_log)
y_pred_log = modelC.predict(sm.add_constant(X_test_c))

# ensure numeric array before exp (fixes earlier TypeError)
y_pred_log_arr = np.asarray(y_pred_log, dtype=float)
y_pred_C = np.exp(y_pred_log_arr)
print_eval(y_test, y_pred_C, label="Model C (log-target)")

# Regularization: RidgeCV & LassoCV
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(X_train_s, y_train)
y_pred_ridge = ridge_cv.predict(X_test_s)
print("Ridge alpha:", ridge_cv.alpha_)
print_eval(y_test, y_pred_ridge, label="RidgeCV")

lasso_cv = LassoCV(cv=5, max_iter=5000, random_state=42).fit(X_train_s, y_train)
y_pred_lasso = lasso_cv.predict(X_test_s)
print("Lasso alpha:", lasso_cv.alpha_)
print_eval(y_test, y_pred_lasso, label="LassoCV")

coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'ridge_coef': ridge_cv.coef_,
    'lasso_coef': lasso_cv.coef_
})
print("\nRidge/Lasso coefficients:\n", coef_df.to_string(index=False))

out_clean = os.path.join(os.path.dirname(DATA_PATH), "ToyotaCorolla_MLR_cleaned.csv")
pd.concat([X, y], axis=1).to_csv(out_clean, index=False)
print("\nSaved cleaned dataset to:", out_clean)
