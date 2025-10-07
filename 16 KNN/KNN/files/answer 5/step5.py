# Step 5: Choosing an appropriate distance metric and value for K

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\16 KNN\KNN\Zoo.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=["animal name"])
X = df.drop(columns=["type"])
y = df["type"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline: scaling + KNN
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Define parameter grid for grid search
param_grid = {
    "knn__n_neighbors": list(range(1, 21)),  # K = 1 to 20
    "knn__p": [1, 2],                        # Manhattan (1) and Euclidean (2)
    "knn__weights": ["uniform", "distance"]  # test both weighting schemes
}

# Perform grid search
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

# Display best results
print("Best parameters:", grid.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(grid.best_score_))

# Evaluate on test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.4f}".format(test_acc))

# Visualize K vs accuracy for each distance metric
results = pd.DataFrame(grid.cv_results_)
plt.figure(figsize=(8, 5))
for p_val, label in zip([1, 2], ['Manhattan (p=1)', 'Euclidean (p=2)']):
    subset = results[results['param_knn__p'] == p_val]
    plt.plot(subset['param_knn__n_neighbors'], subset['mean_test_score'], marker='o', label=label)
plt.xlabel("Number of Neighbours (K)")
plt.ylabel("Mean CV Accuracy")
plt.title("KNN Performance across K values and Distance Metrics")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
