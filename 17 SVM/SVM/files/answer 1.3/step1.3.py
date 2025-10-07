"""
feature_correlations.py
Full pipeline to investigate feature correlations (robust loader + debug prints).
Drop this file into your project and run with your venv python.
"""

import os
import sys
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", font_scale=1.0)

# ---------- User config ----------
CSV_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\mushroom.csv"
TARGET_COL = None
OUTPUT_DIR = r"D:\DATA SCIENCE\ASSIGNMENTS\17 SVM\SVM\correlation_outputs"
DROP_THRESHOLD_NUNIQUE = 1
FILLNA_STRATEGY = "mode"
# ---------------------------------

# quick sanity checks
print("Current working directory:", os.getcwd())
print("CSV_PATH (raw):", CSV_PATH)
print("CSV_PATH (absolute):", os.path.abspath(CSV_PATH))
print("CSV_PATH exists?", os.path.exists(CSV_PATH))
print("Readable by current user?", os.access(os.path.abspath(CSV_PATH), os.R_OK))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Robust load ----------
def try_read_csv(path):
    """Try several common encodings/separators and return (df, used_params) or raise."""
    attempts = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": "\t", "encoding": "utf-8"},
    ]
    last_exc = None
    for params in attempts:
        try:
            df = pd.read_csv(path, **params)
            return df, params
        except Exception as e:
            last_exc = e
    # final fallback: let pandas infer with engine python (slower but forgiving)
    try:
        df = pd.read_csv(path, engine="python")
        return df, {"engine": "python"}
    except Exception as e:
        raise last_exc or e

# If user provided a DataFrame in the environment (rare here), use that
df = globals().get("mushroom_df", None)

if df is None:
    if not os.path.exists(CSV_PATH):
        sys.exit(f"File not found: {os.path.abspath(CSV_PATH)}\nCheck path, spelling, and that the drive is accessible.")
    try:
        df, used = try_read_csv(CSV_PATH)
        print("Loaded CSV successfully with params:", used)
    except PermissionError as pe:
        sys.exit(f"Permission error reading file: {pe}\nCheck file permissions.")
    except Exception as e:
        # show full info to help debugging
        import traceback
        tb = traceback.format_exc()
        sys.exit(f"Failed to read CSV. Last exception:\n{e}\n\nTraceback:\n{tb}")

# Basic confirmation
print("Dataframe shape:", getattr(df, "shape", None))
print("First 5 rows:")
print(df.head().to_string(index=False))
print("\nDataFrame info:")
print(df.info())

# ---------- Basic cleaning ----------
try:
    nunique = df.nunique(dropna=True)
    const_cols = list(nunique[nunique <= DROP_THRESHOLD_NUNIQUE].index)
    if const_cols:
        print(f"Dropping constant / low-variance columns: {const_cols}")
        df = df.drop(columns=const_cols)
except Exception as e:
    print("Warning during dropping constant columns:", e)

# Fill NAs simply (user can adjust)
if FILLNA_STRATEGY == "mode":
    for c in df.columns:
        if df[c].isna().any():
            try:
                df[c].fillna(df[c].mode().iloc[0], inplace=True)
            except Exception:
                df[c].fillna(method="ffill", inplace=True)
elif FILLNA_STRATEGY == "median":
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c].fillna(df[c].median(), inplace=True)

# Automatic dtype coercion for mostly-numeric object columns
for col in df.columns:
    if df[col].dtype == "object":
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() / len(coerced) > 0.6:
            df[col] = coerced

# ---------- Split numeric & categorical ----------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\nNumeric columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# ---------- Helpers ----------
def savefig_and_show(fig, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved figure to {path}")
    plt.close(fig)

def cramers_v(series_x, series_y):
    confusion = pd.crosstab(series_x, series_y)
    if confusion.size == 0:
        return np.nan
    chi2, p, dof, expected = chi2_contingency(confusion)
    n = confusion.sum().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom == 0:
        return 0.0
    return np.sqrt(phi2corr / denom)

def correlation_ratio(categories, measurements):
    categories = pd.Series(categories)
    measurements = pd.Series(measurements)
    mask = categories.notna() & measurements.notna()
    categories = categories[mask]
    measurements = measurements[mask]
    if len(measurements) == 0:
        return np.nan
    cat_groups = measurements.groupby(categories)
    mean_total = measurements.mean()
    ss_between = sum([(grp.size * (grp.mean() - mean_total)**2) for _, grp in cat_groups])
    ss_total = ((measurements - mean_total)**2).sum()
    if ss_total == 0:
        return 0.0
    return np.sqrt(ss_between / ss_total)

# ---------- Numeric correlations ----------
if num_cols:
    pearson = df[num_cols].corr(method="pearson")
    spearman = df[num_cols].corr(method="spearman")
    pearson.to_csv(os.path.join(OUTPUT_DIR, "pearson_correlation_matrix.csv"))
    spearman.to_csv(os.path.join(OUTPUT_DIR, "spearman_correlation_matrix.csv"))
    print("Saved numeric correlation matrices (pearson, spearman).")
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols)*0.5), max(4, len(num_cols)*0.5)))
    sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", square=False,
                cbar_kws={'shrink': .6}, linewidths=.5)
    ax.set_title("Pearson Correlation (Numeric features)")
    savefig_and_show(fig, "pearson_heatmap.png")
    pairs = []
    for a, b in combinations(num_cols, 2):
        pairs.append((a, b, pearson.loc[a, b]))
    top_abs = sorted(pairs, key=lambda x: -abs(x[2]))[:20]
    top_df = pd.DataFrame(top_abs, columns=["feature_a", "feature_b", "pearson_corr"])
    top_df.to_csv(os.path.join(OUTPUT_DIR, "top_numeric_pairs_by_abs_pearson.csv"), index=False)
    print("Saved top numeric correlated pairs.")
else:
    print("No numeric columns found; skipping numeric correlation.")

# ---------- Categorical vs Categorical (Cramér's V) ----------
if len(cat_cols) >= 2:
    cramers_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for a, b in combinations(cat_cols, 2):
        v = cramers_v(df[a], df[b])
        cramers_matrix.loc[a, b] = v
        cramers_matrix.loc[b, a] = v
    np.fill_diagonal(cramers_matrix.values, 1.0)
    cramers_matrix = cramers_matrix.fillna(0.0).astype(float)
    cramers_matrix.to_csv(os.path.join(OUTPUT_DIR, "cramers_v_matrix.csv"))
    print("Saved Cramér's V matrix for categorical features.")
    fig, ax = plt.subplots(figsize=(max(6, len(cat_cols)*0.35), max(6, len(cat_cols)*0.35)))
    sns.heatmap(cramers_matrix, annot=True, fmt=".2f", cmap="vlag", linewidths=.3)

    ax.set_title("Cramér's V (Categorical vs Categorical)")
    savefig_and_show(fig, "cramers_v_heatmap.png")
    cat_pairs = []
    for a, b in combinations(cat_cols, 2):
        cat_pairs.append((a, b, cramers_matrix.loc[a, b]))
    top_cat = sorted(cat_pairs, key=lambda x: -x[2])[:30]
    pd.DataFrame(top_cat, columns=["cat_a", "cat_b", "cramers_v"]).to_csv(
        os.path.join(OUTPUT_DIR, "top_categorical_pairs_by_cramers.csv"), index=False)
    print("Saved top categorical pairs by Cramér's V.")
else:
    print("Not enough categorical columns for Cramér's V (need >=2).")

# ---------- Categorical -> Numeric (Correlation ratio) ----------
if cat_cols and num_cols:
    eta_matrix = pd.DataFrame(index=cat_cols, columns=num_cols, dtype=float)
    for c in cat_cols:
        for n in num_cols:
            eta_matrix.loc[c, n] = correlation_ratio(df[c], df[n])
    eta_matrix = eta_matrix.fillna(0.0).astype(float)
    eta_matrix.to_csv(os.path.join(OUTPUT_DIR, "eta_correlation_ratio_matrix.csv"))
    print("Saved correlation ratio (eta) matrix for categorical->numeric.")
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols)*0.5), max(4, len(cat_cols)*0.25)))
    sns.heatmap(eta_matrix, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.3)
    ax.set_title("Correlation Ratio (categorical -> numeric) η")
    savefig_and_show(fig, "eta_heatmap.png")
    top_eta_rows = []
    for c in cat_cols:
        row = eta_matrix.loc[c].sort_values(ascending=False)[:10]
        for n, val in row.items():
            top_eta_rows.append((c, n, val))
    pd.DataFrame(top_eta_rows, columns=["categorical", "numeric", "eta"]).to_csv(
        os.path.join(OUTPUT_DIR, "top_categorical_to_numeric_eta.csv"), index=False)
    print("Saved categorical -> numeric top explanations (eta).")
else:
    print("Skipping categorical->numeric eta matrix (need both categorical and numeric columns).")

# ---------- Extra: numeric vs binary categorical using point-biserial (if present) ----------
binary_cat = [c for c in cat_cols if df[c].nunique() == 2]
if binary_cat and num_cols:
    pb_list = []
    for c in binary_cat:
        values = pd.Categorical(df[c]).codes
        for n in num_cols:
            try:
                r, p = pointbiserialr(values, df[n])
                pb_list.append((c, n, r, p))
            except Exception:
                pb_list.append((c, n, np.nan, np.nan))
    pb_df = pd.DataFrame(pb_list, columns=["binary_cat", "numeric", "pointbiserial_r", "p_value"])
    pb_df.to_csv(os.path.join(OUTPUT_DIR, "pointbiserial_binary_cat_numeric.csv"), index=False)
    print("Saved point-biserial correlations for binary categorical features.")
else:
    print("No binary categorical columns or no numeric columns found; skipping point-biserial step.")

# ---------- Summary output: top correlations consolidated ----------
summary_rows = []
if num_cols:
    for _, r in top_df.iterrows():
        summary_rows.append({
            "type": "numeric-numeric",
            "a": r['feature_a'],
            "b": r['feature_b'],
            "score": r['pearson_corr']
        })
if len(cat_cols) >= 2:
    for row in top_cat:
        summary_rows.append({
            "type": "cat-cat",
            "a": row[0],
            "b": row[1],
            "score": row[2]
        })
if cat_cols and num_cols:
    for c, n, val in top_eta_rows:
        summary_rows.append({
            "type": "cat->num",
            "a": c,
            "b": n,
            "score": val
        })
summary_df = pd.DataFrame(summary_rows).sort_values(by="score", key=lambda col: col.abs(), ascending=False)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "consolidated_top_correlations.csv"), index=False)
print(f"Saved consolidated top correlations to {os.path.join(OUTPUT_DIR, 'consolidated_top_correlations.csv')}")
print("\nDONE — All outputs are in the folder:", OUTPUT_DIR)
