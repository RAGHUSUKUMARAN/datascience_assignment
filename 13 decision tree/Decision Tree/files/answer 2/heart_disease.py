# decision_tree_assignment.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# === PATHS ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\13 decision tree\Decision Tree"
file_path = os.path.join(base_path, "heart_disease.xlsx")

# === LOAD DATA ===
df = pd.read_excel(file_path, sheet_name="Heart_disease")

# === CLEANING ===
# Convert boolean to int
for col in ['fbs', 'exang']:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
    elif df[col].dtype == object:
        df[col] = df[col].map({'True':1, 'TURE':1, 'FALSE':0, 'False':0})  # fix odd strings

# Sex mapping
df['sex'] = df['sex'].map({'Male':1, 'Female':0})

# Target binary (num>0 → 1)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# One-hot encode categorical vars
ohe_cols = ['cp','restecg','slope','thal']
df_processed = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# Drop original num column
df_processed = df_processed.drop(columns=['num'])

# === TRAIN/TEST SPLIT ===
X = df_processed.drop(columns=['target'])
y = df_processed['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === BASELINE MODEL ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

# === EVALUATION ===
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_proba)
}

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# === SAVE RESULTS ===
# Processed dataset
processed_path = os.path.join(base_path, "heart_processed.csv")
df_processed.to_csv(processed_path, index=False)

# Model
model_path = os.path.join(base_path, "decision_tree_model.pkl")
joblib.dump(clf, model_path)

# Metrics report
report_path = os.path.join(base_path, "decision_tree_report.txt")
with open(report_path, "w") as f:
    f.write("=== Decision Tree Evaluation ===\n")
    for k,v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))

# Confusion matrix plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_path = os.path.join(base_path, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# Decision Tree visualization
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["No Disease","Disease"],
          filled=True, rounded=True, fontsize=8)
tree_path = os.path.join(base_path, "decision_tree.png")
plt.savefig(tree_path)
plt.close()

print("All done! Files saved in:", base_path)
print("Processed CSV:", processed_path)
print("Model:", model_path)
print("Report:", report_path)
print("Plots:", cm_path, "and", tree_path)
