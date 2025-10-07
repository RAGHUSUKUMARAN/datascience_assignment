# Step 2: Frequent Itemsets (single items) — support >= 5%
# Works with either:
#  - transactions (list of lists) from preprocessing, OR
#  - one-hot CSV saved from preprocessing

import pandas as pd
from collections import Counter
from typing import List

# ---------- Config ----------
MIN_SUPPORT = 0.05          # 5%
total_transactions = None  # filled later
# Input paths (change if needed)
one_hot_csv_path = r"D:\DATA SCIENCE\ASSIGNMENTS\10 association rules\Association Rules\basket_one_hot.csv"
# If you have the transactions list in memory (from preprocessing), set it here:
# transactions = [...]   # list of lists, each inner list = items (lowercased)
# ----------------------------

def frequent_items_from_transactions(transactions: List[List[str]], min_support: float):
    """
    Count single-item supports from a list-of-lists transactions and return items >= min_support.
    Returns a DataFrame: item | support | count
    """
    n = len(transactions)
    total_transactions_local = n
    # Count occurrences (in how many transactions each item appears)
    counter = Counter()
    for tx in transactions:
        unique_items = set(tx)  # ensure a transaction counts an item only once
        counter.update(unique_items)
    rows = []
    for item, count in counter.items():
        support = count / total_transactions_local
        rows.append((item, support, count))
    df = pd.DataFrame(rows, columns=["item", "support", "count"]).sort_values("support", ascending=False).reset_index(drop=True)
    df = df[df["support"] >= min_support]
    return df, total_transactions_local

def frequent_items_from_onehot(one_hot_df: pd.DataFrame, min_support: float):
    """
    Compute supports directly from one-hot DataFrame where rows = transactions, cols = items (0/1).
    """
    n = one_hot_df.shape[0]
    support_series = one_hot_df.sum(axis=0) / n
    counts = (one_hot_df.sum(axis=0)).astype(int)
    df = pd.DataFrame({
        "item": support_series.index,
        "support": support_series.values,
        "count": counts.values
    }).sort_values("support", ascending=False).reset_index(drop=True)
    df = df[df["support"] >= min_support]
    return df, n

# -------------------------
# Try to read one-hot CSV first; fallback to transactions variable if not available
# -------------------------
import os

if os.path.exists(one_hot_csv_path):
    print("Loading one-hot CSV from:", one_hot_csv_path)
    onehot = pd.read_csv(one_hot_csv_path, index_col=False)  # columns are items
    # If your CSV saved with an index column, you may need index_col=0 — adjust if necessary.
    # Ensure values are 0/1:
    onehot = onehot.fillna(0).astype(int)
    freq_df, total_transactions = frequent_items_from_onehot(onehot, MIN_SUPPORT)
else:
    print("One-hot CSV not found at the path. Looking for 'transactions' variable in memory...")
    try:
        # Use the transactions list produced earlier in preprocessing step
        transactions  # type: ignore
        freq_df, total_transactions = frequent_items_from_transactions(transactions, MIN_SUPPORT)  # type: ignore
    except NameError:
        raise RuntimeError("No input found: place 'transactions' variable in memory or save one-hot CSV at one_hot_csv_path.")

# Pretty print top results
print(f"\nTotal transactions used: {total_transactions}")
print(f"Items with support >= {MIN_SUPPORT*100:.1f}% (count >= {int(MIN_SUPPORT*total_transactions)}):\n")
pd.set_option("display.max_rows", None)
print(freq_df.to_string(index=False))

# Save the frequent single-item list to CSV for embedding in your assignment
out_csv = r"D:\DATA SCIENCE\ASSIGNMENTS\10 association rules\Association Rules\frequent_items_single.csv"
freq_df.to_csv(out_csv, index=False)
print(f"\nSaved frequent single-item list to: {out_csv}")

# Quick tips:
# - If you want the top-k items only, do freq_df.head(k)
# - To change threshold, modify MIN_SUPPORT variable at the top
