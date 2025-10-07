# naive_bayes_text_mining.py
"""
Text classification using Multinomial Naive Bayes + sentiment analysis (VADER)
Adapt this INPUT_PATH if needed.
Saves outputs to the same folder as INPUT_PATH.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Sentiment (VADER)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------
# Config - change this path if needed
# -----------------------
INPUT_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\19 naive bayes and text mining\blogs.csv"
OUTPUT_FOLDER = os.path.dirname(INPUT_PATH)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# TF-IDF / model settings
MAX_FEATURES = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2
GRID = {
    # small grid to tune key hyperparameters quickly
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__alpha": [0.1, 0.5, 1.0]
}
CV = 4

# -----------------------
# Helpers
# -----------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str):
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

def save_confusion_matrix(cm, labels, path_png, title="Confusion matrix"):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()

# -----------------------
# Load & preprocess
# -----------------------
print("Loading:", INPUT_PATH)
df = pd.read_csv(INPUT_PATH)

# detect columns (flexible)
text_col = None
label_col = None
for c in ["Data","Text","data","text","Content"]:
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
        raise ValueError("Could not find text/label columns in CSV.")

df = df[[text_col, label_col]].rename(columns={text_col: "Data", label_col: "Labels"})
df = df.dropna(subset=["Data", "Labels"]).reset_index(drop=True)
print("Rows after dropna:", len(df))
print("Labels distribution:\n", df["Labels"].value_counts().head(20))

# Clean text
df["clean_text"] = df["Data"].astype(str).apply(clean_text)
df["clean_text_nostop"] = df["clean_text"].apply(remove_stopwords)
df["clean_len_words"] = df["clean_text_nostop"].apply(lambda t: len(t.split()))

# Save processed CSV
processed_csv = os.path.join(OUTPUT_FOLDER, "blogs_processed_naivebayes.csv")
df.to_csv(processed_csv, index=False)
print("Saved processed CSV:", processed_csv)

# -----------------------
# TF-IDF + train/test
# -----------------------
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X = vectorizer.fit_transform(df["clean_text_nostop"].fillna(""))
joblib.dump(vectorizer, os.path.join(OUTPUT_FOLDER, "tfidf_vectorizer.joblib"))
print("TF-IDF shape:", X.shape)

le = LabelEncoder()
y = le.fit_transform(df["Labels"])
joblib.dump(le, os.path.join(OUTPUT_FOLDER, "label_encoder.joblib"))
print("Classes:", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print("Train/test sizes:", X_train.shape, X_test.shape)

# -----------------------
# Baseline MultinomialNB
# -----------------------
print("\nTraining baseline MultinomialNB...")
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test) if hasattr(nb, "predict_proba") else None

acc = accuracy_score(y_test, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
print(f"Baseline accuracy: {acc:.4f}, f1_macro: {f1_macro:.4f}")

# Save baseline model
joblib.dump(nb, os.path.join(OUTPUT_FOLDER, "nb_baseline.joblib"))

# Save baseline metrics & reports
base_report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
with open(os.path.join(OUTPUT_FOLDER, "baseline_classification_report.txt"), "w") as f:
    f.write(base_report)
pd.DataFrame(confusion_matrix(y_test, y_pred), index=le.classes_, columns=le.classes_).to_csv(os.path.join(OUTPUT_FOLDER, "baseline_confusion_matrix.csv"))
save_confusion_matrix(confusion_matrix(y_test, y_pred), le.classes_, os.path.join(OUTPUT_FOLDER, "baseline_confusion_matrix.png"))

# Save baseline predictions
pred_df = pd.DataFrame({
    "text": df.loc[X_test.indices if hasattr(X_test, 'indices') else X_test.tolist(), "Data"].values if False else df.iloc[X_test.nonzero()[0]]["Data"].values,  # placeholder not used
})
# Better approach: map test indices
test_idx = X_test.nonzero()[0] if hasattr(X_test, "nonzero") else None
# We'll use index-based split to save predictions accurately:
_, X_test_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
predictions_df = pd.DataFrame({
    "true_label": le.inverse_transform(y_test),
    "pred_label": le.inverse_transform(y_pred),
    "pred_confidence": y_prob.max(axis=1) if y_prob is not None else None,
    "text": df.loc[X_test_idx, "Data"].values
})
predictions_df.to_csv(os.path.join(OUTPUT_FOLDER, "baseline_predictions.csv"), index=False)

# -----------------------
# Quick GridSearch (pipeline) to tune alpha + ngram_range
# -----------------------
print("\nStarting small GridSearch over alpha / ngram_range...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, min_df=2, sublinear_tf=True)),
    ("clf", MultinomialNB())
])
grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__alpha": [0.1, 0.5, 1.0]
}
gs = GridSearchCV(pipeline, grid, cv=CV, n_jobs=-1, verbose=1, scoring="accuracy")
# Fit on raw cleaned text (pipeline will vectorize)
gs.fit(df.loc[:, "clean_text_nostop"], y)
print("GridSearch best:", gs.best_params_, "best_score:", gs.best_score_)

# Evaluate best estimator on held-out test set
best_model = gs.best_estimator_
y_pred_gs = best_model.predict(df.loc[X_test_idx, "clean_text_nostop"])
acc_gs = accuracy_score(y_test, y_pred_gs)
print(f"Tuned model accuracy on test set: {acc_gs:.4f}")

# Save tuned pipeline
joblib.dump(best_model, os.path.join(OUTPUT_FOLDER, "nb_tuned_pipeline.joblib"))

# Save tuned reports
with open(os.path.join(OUTPUT_FOLDER, "tuned_classification_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_gs, target_names=le.classes_, digits=4))
pd.DataFrame(confusion_matrix(y_test, y_pred_gs), index=le.classes_, columns=le.classes_).to_csv(os.path.join(OUTPUT_FOLDER, "tuned_confusion_matrix.csv"))
save_confusion_matrix(confusion_matrix(y_test, y_pred_gs), le.classes_, os.path.join(OUTPUT_FOLDER, "tuned_confusion_matrix.png"))

# -----------------------
# Sentiment analysis using VADER
# -----------------------
print("\nRunning VADER sentiment analysis...")
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# compute sentiment scores
sent_scores = df["Data"].astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
def sentiment_label(c):
    if c >= 0.05:
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"
df["sentiment_score"] = sent_scores
df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

# Save CSV with sentiment + predictions (merge predictions_df on text)
# Attach baseline predictions where possible by index:
# We already saved predictions_df for the test subset; let's save overall sentiment + labels for full data.
df.to_csv(os.path.join(OUTPUT_FOLDER, "blogs_with_sentiment.csv"), index=False)
print("Saved sentiment-annotated CSV to:", os.path.join(OUTPUT_FOLDER, "blogs_with_sentiment.csv"))

# -----------------------
# Summary JSON for assignment
# -----------------------
summary = {
    "n_documents": int(len(df)),
    "n_classes": int(len(le.classes_)),
    "classes": list(le.classes_),
    "baseline_accuracy": float(acc),
    "tuned_grid_best": gs.best_params_,
    "tuned_cv_score": float(gs.best_score_),
    "tuned_test_accuracy": float(acc_gs)
}
with open(os.path.join(OUTPUT_FOLDER, "nb_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll done. Outputs saved to:", OUTPUT_FOLDER)
print("Key files:")
print(" - baseline_classification_report.txt")
print(" - baseline_confusion_matrix.csv/png")
print(" - baseline_predictions.csv")
print(" - nb_baseline.joblib")
print(" - nb_tuned_pipeline.joblib (GridSearch best)")
print(" - tuned_classification_report.txt")
print(" - blogs_with_sentiment.csv")
print(" - nb_summary.json")
