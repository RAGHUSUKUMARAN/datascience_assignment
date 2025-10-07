# train_and_save.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Example dataset (swap with your CSV)
data = load_breast_cancer(as_frame=True)
X = data.frame[data.feature_names]
y = data.target

# Train-test quick split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler and logistic regression
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_s, y_train)

# Save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Saved model.joblib and scaler.joblib")
print("Feature order (must match at prediction time):")
print(list(X.columns))
