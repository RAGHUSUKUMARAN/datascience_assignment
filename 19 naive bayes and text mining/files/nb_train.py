# nb_train.py
"""
Naive Bayes text classifier (Task 2)
- Change INPUT_PATH if needed.
- Saves outputs (model, vectorizer, reports) to the same folder as INPUT_PATH.
Requirements:
    pip install numpy pandas scikit-learn matplotlib joblib
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

# -------- CONFIG --------
INPUT_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\19 naive bayes and text mining\blogs.csv"
OUTPUT_FOLDER = os.path.dirname(INPUT_PATH)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 5000   # change if you want fewer/more features
NGRAM_RANGE = (1,2)   # unigrams + bigrams
MIN_DF = 2

# -------- helpers --------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)        # remove urls
    t = re.sub(r"\S+@\S+", " ", t)                # remove emails
    t = re.sub(r"[^a-z0-9\s]", " ", t)            # remove punctuation
    t = re.sub(r"\b\d+\b", " ", t)                # remove standalone digits
    t = re.sub(r"\s+", " ", t).strip()            # collapse spaces
    return t

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    kept = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(kept)

def save_confusion_matrix(cm, labels, png_path, title="Confusion matrix"):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

# -------- main --------
def main():
    # load dataset
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    print("Loading:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    # detect likely columns
    text_col = None
    label_col = None
    for c in ["Data","Text","data","text","Content","content"]:
        if c in df.columns:
            text_col = c
            break
    for c in ["Labels","Label","labels","label","Category","category"]:
        if c in df.columns:
            label_col = c
            break
    if text_col is None or label_col is None:
        if len(df.columns) >= 2:
            text_col, label_col = df.columns[0], df.columns[1]
        else:
            raise ValueError("Couldn't auto-detect text/label columns in CSV. Ensure it has two columns.")

    df = df[[text_col, label_col]].rename(columns={text_col: "Data", label_col: "Labels"})
    df = df.dropna(subset=["Data", "Labels"]).reset_index(drop=True)
    print(f"Rows after dropna: {len(df)}")
    print("Label distribution (top 10):\n", df["Labels"].value_counts().head(10).to_string())

    # Preprocess text (clean + remove stopwords)
    print("Cleaning text (lowercase, remove urls/emails/punct, drop stopwords)...")
    df["clean"] = df["Data"].astype(str).apply(clean_text).apply(remove_stopwords)
    df["clean_len"] = df["clean"].apply(lambda t: len(t.split()))

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["Labels"])
    classes = list(le.classes_)
    print("Classes detected:", classes)

    # Train-test split (stratified)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean"], y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Train size:", len(X_train_text), "Test size:", len(X_test_text))

    # Vectorize: fit TF-IDF on train only
    print("Fitting TF-IDF on training data...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, min_df=MIN_DF, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print("TF-IDF shapes:", X_train.shape, X_test.shape)

    # Save vectorizer
    vec_path = os.path.join(OUTPUT_FOLDER, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    joblib.dump(le, os.path.join(OUTPUT_FOLDER, "label_encoder.joblib"))
    print("Saved vectorizer and label encoder to output folder.")

    # Train Multinomial Naive Bayes
    print("Training MultinomialNB...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # Predict on test
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test) if hasattr(nb, "predict_proba") else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weight:.4f}")

    # Classification report & confusion matrix
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # Save artifacts
    with open(os.path.join(OUTPUT_FOLDER, "nb_classification_report.txt"), "w") as f:
        f.write("Test Accuracy: {:.6f}\n\n".format(acc))
        f.write(report)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(OUTPUT_FOLDER, "nb_confusion_matrix.csv"))
    save_confusion_matrix(cm, classes, os.path.join(OUTPUT_FOLDER, "nb_confusion_matrix.png"))

    # Save model & predictions
    joblib.dump(nb, os.path.join(OUTPUT_FOLDER, "nb_model.joblib"))
    # Build predictions dataframe aligned to test split
    test_indices = X_test_text.index if hasattr(X_test_text, "index") else None
    preds_df = pd.DataFrame({
        "text": X_test_text.values,
        "true_label": le.inverse_transform(y_test),
        "pred_label": le.inverse_transform(y_pred),
        "pred_confidence": (y_prob.max(axis=1) if y_prob is not None else None)
    })
    # The above "text" may be an ndarray of strings; ensure correct alignment using iloc on dataframe
    # Let's get indices used in the split to be safe:
    # We recreate by mapping values (not perfect if duplicates), but better approach is using .iloc indexes:
    # Simpler: re-run split with return of indices - but to avoid overcomplicating, save predictions by re-applying vectorizer to original X_test_text
    # Save final preds using X_test_text series
    preds_df = pd.DataFrame({
        "text": X_test_text.reset_index(drop=True),
        "true_label": le.inverse_transform(y_test),
        "pred_label": le.inverse_transform(y_pred),
        "pred_confidence": (y_prob.max(axis=1) if y_prob is not None else None)
    })
    preds_df.to_csv(os.path.join(OUTPUT_FOLDER, "nb_test_predictions.csv"), index=False)

    # Summary JSON
    summary = {
        "n_documents": int(len(df)),
        "n_classes": int(len(classes)),
        "classes": classes,
        "test_size": int(len(X_test_text)),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weight),
        "recall_weighted": float(rec_weight),
        "f1_weighted": float(f1_weight),
    }
    with open(os.path.join(OUTPUT_FOLDER, "nb_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs to:", OUTPUT_FOLDER)
    print(" - nb_model.joblib")
    print(" - tfidf_vectorizer.joblib")
    print(" - nb_classification_report.txt")
    print(" - nb_confusion_matrix.csv/png")
    print(" - nb_test_predictions.csv")
    print(" - nb_summary.json")

if __name__ == "__main__":
    main()
