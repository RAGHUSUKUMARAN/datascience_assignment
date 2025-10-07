"""
Step 2b: Pairwise frequent itemsets + association rules (support, confidence, lift)
- Works from one-hot DataFrame (preferred) or transactions list.
- Only considers pairs made from single items that meet min_support (Apriori pruning).
"""

import pandas as pd
from itertools import combinations
from typing import List, Tuple
import os

# -------------------- Config --------------------
MIN_SUPPORT = 0.05          # minimum support threshold (fraction of transactions)
MIN_CONFIDENCE = 0.0       # optional: filter rules with confidence >= this (0 to 1). Set 0 to keep all.
ONE_HOT_CSV = r"D:\DATA SCIENCE\ASSIGNMENTS\10 association rules\Association Rules\basket_one_hot.csv"
OUT_RULES_CSV = r"D:\DATA SCIENCE\ASSIGNMENTS\10 association rules\Association Rules\pairwise_rules.csv"
# ------------------------------------------------

def load_onehot_or_transactions(one_hot_path: str):
    """
    Try loading one-hot CSV; otherwise expect `transactions` list to exist in memory.
    Returns: (onehot_df, transactions_list)
    One of them may be None depending on source.
    """
    if os.path.exists(one_hot_path):
        onehot = pd.read_csv(one_hot_path, index_col=False)
        # Ensure binary 0/1
        onehot = onehot.fillna(0).astype(int)
        return onehot, None
    else:
        try:
            # transactions variable should be a list of lists (from preprocessing)
            transactions  # type: ignore
            return None, transactions  # type: ignore
        except NameError:
            raise RuntimeError("No input found. Provide basket_one_hot.csv at ONE_HOT_CSV path or ensure 'transactions' exists in memory.")

def compute_single_item_supports_from_onehot(onehot: pd.DataFrame) -> Tuple[pd.Series, int]:
    n = onehot.shape[0]
    support = onehot.sum(axis=0) / n
    counts = onehot.sum(axis=0).astype(int)
    return pd.DataFrame({"support": support, "count": counts}), n

def compute_single_item_supports_from_tx(transactions: List[List[str]]) -> Tuple[pd.DataFrame, int]:
    from collections import Counter
    n = len(transactions)
    cnt = Counter()
    for tx in transactions:
        cnt.update(set(tx))
    items = []
    counts = []
    supports = []
    for itm, c in cnt.items():
        items.append(itm)
        counts.append(c)
        supports.append(c / n)
    df = pd.DataFrame({"item": items, "support": supports, "count": counts}).set_index("item")
    return df, n

def generate_pairwise_rules_from_onehot(onehot: pd.DataFrame, min_support: float, min_confidence: float):
    single_df, n = compute_single_item_supports_from_onehot(onehot)
    # Keep items that meet min_support (Apriori)
    frequent_items = single_df[single_df["support"] >= min_support].copy()
    frequent_items = frequent_items.sort_values("support", ascending=False)
    items = frequent_items.index.tolist()

    rules = []
    for a, b in combinations(items, 2):
        # pair support = count of transactions where both a and b = 1
        pair_count = ((onehot[a] == 1) & (onehot[b] == 1)).sum()
        pair_support = pair_count / n
        if pair_support >= min_support:
            support_a = frequent_items.loc[a, "support"]
            support_b = frequent_items.loc[b, "support"]
            # confidence a->b and b->a
            conf_a_b = pair_support / support_a if support_a > 0 else 0.0
            conf_b_a = pair_support / support_b if support_b > 0 else 0.0
            lift = pair_support / (support_a * support_b) if (support_a * support_b) > 0 else 0.0

            if conf_a_b >= min_confidence:
                rules.append({
                    "antecedent": a,
                    "consequent": b,
                    "support": pair_support,
                    "confidence": conf_a_b,
                    "lift": lift,
                    "pair_count": int(pair_count)
                })
            if conf_b_a >= min_confidence:
                rules.append({
                    "antecedent": b,
                    "consequent": a,
                    "support": pair_support,
                    "confidence": conf_b_a,
                    "lift": lift,
                    "pair_count": int(pair_count)
                })

    rules_df = pd.DataFrame(rules).sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)
    return rules_df, frequent_items, n

def generate_pairwise_rules_from_tx(transactions: List[List[str]], min_support: float, min_confidence: float):
    # Build a quick mapping of item -> set of transaction indices
    item_to_tids = {}
    for tid, tx in enumerate(transactions):
        for itm in set(tx):
            item_to_tids.setdefault(itm, set()).add(tid)
    n = len(transactions)
    # single support
    single_support = {itm: len(tids)/n for itm, tids in item_to_tids.items()}
    # frequent items
    frequent_items = [itm for itm, sup in single_support.items() if sup >= min_support]
    rules = []
    for a, b in combinations(sorted(frequent_items), 2):
        tids_a = item_to_tids[a]
        tids_b = item_to_tids[b]
        pair_tids = tids_a & tids_b
        pair_count = len(pair_tids)
        pair_support = pair_count / n
        if pair_support >= min_support:
            support_a = single_support[a]
            support_b = single_support[b]
            conf_a_b = pair_support / support_a if support_a > 0 else 0.0
            conf_b_a = pair_support / support_b if support_b > 0 else 0.0
            lift = pair_support / (support_a * support_b) if (support_a * support_b) > 0 else 0.0
            if conf_a_b >= min_confidence:
                rules.append({"antecedent": a, "consequent": b, "support": pair_support, "confidence": conf_a_b, "lift": lift, "pair_count": pair_count})
            if conf_b_a >= min_confidence:
                rules.append({"antecedent": b, "consequent": a, "support": pair_support, "confidence": conf_b_a, "lift": lift, "pair_count": pair_count})
    rules_df = pd.DataFrame(rules).sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)
    # Convert frequent_items to DataFrame for convenience
    freq_df = pd.DataFrame([(itm, int(len(item_to_tids[itm])), single_support[itm]) for itm in sorted(frequent_items)],
                           columns=["item", "count", "support"]).set_index("item").sort_values("support", ascending=False)
    return rules_df, freq_df, n

# -------------------- Run --------------------
onehot, transactions = load_onehot_or_transactions(ONE_HOT_CSV)

if onehot is not None:
    print("Using one-hot CSV as input.")
    rules_df, freq_items_df, total_tx = generate_pairwise_rules_from_onehot(onehot, MIN_SUPPORT, MIN_CONFIDENCE)
else:
    print("Using 'transactions' variable in memory as input.")
    rules_df, freq_items_df, total_tx = generate_pairwise_rules_from_tx(transactions, MIN_SUPPORT, MIN_CONFIDENCE)

print(f"Total transactions: {total_tx}")
print(f"Frequent single items (support >= {MIN_SUPPORT}): {len(freq_items_df)}\n")
print("Top 10 pairwise rules by lift:")
pd.set_option("display.max_rows", 20)
print(rules_df.head(10).to_string(index=False))

# Save to CSV
rules_df.to_csv(OUT_RULES_CSV, index=False)
print(f"\nSaved pairwise rules to: {OUT_RULES_CSV}")
