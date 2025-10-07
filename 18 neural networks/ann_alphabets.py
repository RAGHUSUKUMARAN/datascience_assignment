"""
ANN for Alphabets classification (basic implementation)

- Expects Alphabets_data.csv available at input_path (change if needed)
- Saves outputs (model, metrics, plots, preds) to output_folder
- Uses TensorFlow / Keras

Requirements:
    pip install numpy pandas scikit-learn matplotlib tensorflow
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------
# Paths - change if needed
# -----------------------
# Input CSV
input_path = r"D:\DATA SCIENCE\ASSIGNMENTS\18 neural networks\Neural networks\Alphabets_data.csv"

# Output folder
output_folder = r"D:\DATA SCIENCE\ASSIGNMENTS\18 neural networks\Neural networks"
os.makedirs(output_folder, exist_ok=True)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv(input_path)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------
# Preprocessing
# -----------------------
target_col = "letter"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in CSV.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Label encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_int = le.fit_transform(y)
num_classes = len(le.classes_)
y_cat = tf.keras.utils.to_categorical(y_int, num_classes=num_classes)
print(f"Detected {num_classes} classes: {list(le.classes_)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_int
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Save preprocessing objects
import joblib
joblib.dump(scaler, os.path.join(output_folder, "scaler.joblib"))
joblib.dump(le, os.path.join(output_folder, "label_encoder.joblib"))

# -----------------------
# Build ANN model
# -----------------------
input_dim = X_train.shape[1]
hidden_units = 128
dropout_rate = 0.2

def build_basic_ann(input_dim, hidden_units=128, dropout_rate=0.2, num_classes=26):
    model = Sequential([
        Dense(hidden_units, input_dim=input_dim, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax")
    ])
    return model

model = build_basic_ann(input_dim, hidden_units, dropout_rate, num_classes)
model.summary()

# -----------------------
# Compile model
# -----------------------
learning_rate = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_path = os.path.join(output_folder, "best_ann_model.h5")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
]

# -----------------------
# Train
# -----------------------
batch_size = 64
epochs = 80

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=2
)

final_model_path = os.path.join(output_folder, "final_ann_model.h5")
model.save(final_model_path)
print(f"Saved model to: {final_model_path} (best checkpoint at {checkpoint_path})")

# -----------------------
# Plot training history
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("epoch"); plt.title("Loss"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("epoch"); plt.title("Accuracy"); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "training_history.png"))
plt.close()
print("Saved training history plot to training_history.png")

# -----------------------
# Evaluate
# -----------------------
y_pred_prob = model.predict(X_test)
y_pred_int = np.argmax(y_pred_prob, axis=1)
y_true_int = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true_int, y_pred_int)
print(f"\nTest accuracy: {acc:.4f}")

report = classification_report(y_true_int, y_pred_int, target_names=le.classes_, digits=4)
print("\nClassification Report:\n", report)

with open(os.path.join(output_folder, "classification_report.txt"), "w") as f:
    f.write(f"Test accuracy: {acc:.4f}\n\n")
    f.write(report)

cm = confusion_matrix(y_true_int, y_pred_int)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
cm_df.to_csv(os.path.join(output_folder, "confusion_matrix.csv"))

plt.figure(figsize=(12,10))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(le.classes_)), le.classes_, rotation=90)
plt.yticks(range(len(le.classes_)), le.classes_)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
plt.close()
print("Saved confusion matrix files")

# -----------------------
# Save predictions
# -----------------------
pred_df = pd.DataFrame({
    "true_label": le.inverse_transform(y_true_int),
    "pred_label": le.inverse_transform(y_pred_int),
    "pred_confidence": np.max(y_pred_prob, axis=1)
})
pred_df.to_csv(os.path.join(output_folder, "test_predictions.csv"), index=False)
print("Saved test predictions to test_predictions.csv")

print("\n✅ All outputs saved in:", output_folder)
