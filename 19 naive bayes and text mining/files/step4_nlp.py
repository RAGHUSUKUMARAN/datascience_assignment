# step4_nlp.py
"""
Evaluates Naive Bayes Text Classifier on blogs.csv and performs evaluation artifacts.
"""

import os, json, re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

# -------- File Paths --------
INPUT_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\19 naive bayes and text mining\blogs.csv"
OUTPUT_PATH = os.path.dirname(INPUT_PATH)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# -------- Preprocessing --------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

stopwords = set(ENGLISH_STOP_WORDS)
def remove_stopwords(t: str) -> str:
    return " ".join([w for w in t.split() if w not in stopwords])

df = pd.read_csv(INPUT_PATH)
text_col = "Data" if "Data" in df.columns else df.columns[0]
label_col = "Labels" if "Labels" in df.columns else df.columns[1]
df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
df["clean"] = df[text_col].apply(clean_text).apply(remove_stopwords)

le = LabelEncoder()
y = le.fit_transform(df[label_col])
classes = list(le.classes_)  # <— used later

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], y, test_size=0.2, stratify=y, random_state=42
)

# -------- Baseline NB --------
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train_vec = vec.fit_transform(X_train)
X_test_vec  = vec.transform(X_test)

nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_vec, y_train)
y_pred = nb.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

print(f"Baseline Accuracy: {acc:.4f}")
print(f"Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weight:.4f}")

# -------- GridSearch Tuning --------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, min_df=2, sublinear_tf=True)),
    ("clf", MultinomialNB())
])
params = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__alpha": [0.1, 0.5, 1.0]
}
gs = GridSearchCV(pipeline, params, cv=4, n_jobs=-1, scoring="accuracy")
gs.fit(X_train, y_train)
y_pred_gs = gs.predict(X_test)

acc_gs = accuracy_score(y_test, y_pred_gs)
f1_macro_gs = precision_recall_fscore_support(y_test, y_pred_gs, average="macro", zero_division=0)[2]
f1_w_gs     = precision_recall_fscore_support(y_test, y_pred_gs, average="weighted", zero_division=0)[2]

print(f"Tuned Accuracy: {acc_gs:.4f} | Macro F1: {f1_macro_gs:.4f}")
print("Best Params:", gs.best_params_)

# -------- Save Results --------
summary = {
    "baseline": {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weight},
    "tuned":    {"accuracy": acc_gs, "f1_macro": f1_macro_gs, "f1_weighted": f1_w_gs, "best_params": gs.best_params_}
}
with open(os.path.join(OUTPUT_PATH, "nb_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Results saved to nb_results.json")

# ----------------------------
# Visualization and Comparison
# ----------------------------
import matplotlib.pyplot as plt

# Confusion matrix from the **tuned** predictions (use baseline if you prefer)
cm = confusion_matrix(y_test, y_pred_gs)

# --- Confusion Matrix Plot ---
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix — Naive Bayes")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=90)
plt.yticks(range(len(classes)), classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
confusion_path = os.path.join(OUTPUT_PATH, "nb_confusion_matrix.png")
plt.savefig(confusion_path, dpi=300)
plt.close()
print(f"Saved confusion matrix plot to: {confusion_path}")

# Also save CSV version of the confusion matrix (handy for the appendix)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    os.path.join(OUTPUT_PATH, "nb_confusion_matrix.csv"), index=True
)

# --- Baseline vs Tuned Comparison ---
baseline_metrics = {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weight}
tuned_metrics    = {"accuracy": acc_gs, "f1_macro": f1_macro_gs, "f1_weighted": f1_w_gs}

comp_df = pd.DataFrame([
    {"model": "baseline", **baseline_metrics},
    {"model": "tuned",    **tuned_metrics}
])

comparison_path = os.path.join(OUTPUT_PATH, "baseline_vs_tuned_metrics.csv")
comp_df.to_csv(comparison_path, index=False)
print(f"Saved metric comparison to: {comparison_path}")
print("\nComparison Table:\n", comp_df)
