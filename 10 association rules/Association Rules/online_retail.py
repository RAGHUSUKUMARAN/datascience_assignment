# Step 1: Data Preprocessing for Association Rule Mining

import pandas as pd

def load_transactions_from_file(path: str, col_index: int = 0):
    """
    Load Excel/CSV file and return a list of transactions (list of lists).
    Each transaction = list of items.
    """
    if path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    
    raw_col = df.iloc[:, col_index].astype(str)

    transactions = []
    for basket_str in raw_col:
        # Split by comma, strip spaces, lowercase
        items = [itm.strip().lower() for itm in basket_str.split(',') if itm.strip()]
        # Remove duplicates inside a basket
        items = list(dict.fromkeys(items))  
        transactions.append(items)

    return transactions

def transactions_to_ohe(transactions):
    """
    Convert list-of-lists (transactions) to one-hot encoded DataFrame.
    """
    tx_df = pd.DataFrame({'tid': range(len(transactions)), 'items': transactions})
    tx_exploded = tx_df.explode('items').dropna(subset=['items'])
    ohe = pd.crosstab(tx_exploded['tid'], tx_exploded['items'])
    ohe = ohe.reindex(range(len(transactions)), fill_value=0)  # keep all transactions
    return ohe

# -------------------------
# Run preprocessing
# -------------------------
filepath = r"D:\DATA SCIENCE\ASSIGNMENTS\10 association rules\Association Rules\Online Retail.xlsx"

transactions = load_transactions_from_file(filepath)
print(f"Total transactions loaded: {len(transactions)}")

# Quick peek at first 5 transactions
for i, t in enumerate(transactions[:5]):
    print(f"{i}: {t}")

basket_ohe = transactions_to_ohe(transactions)
print("\nOne-hot encoded basket shape:", basket_ohe.shape)
print("\nFirst 5 rows:")
print(basket_ohe.head())

# Save the one-hot encoded dataset if needed
basket_ohe.to_csv(r"D:\DATA SCIENCE\ASSIGNMENTS\10 association rules\Association Rules\basket_one_hot.csv", index=False)
print("\nSaved one-hot file to basket_one_hot.csv")
