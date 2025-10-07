# step3_neural.py
"""
ANN for Alphabets classification (baseline + random-search hyperparameter tuning + evaluation report)

- Expects Alphabets_data.csv available at INPUT_PATH (change if needed)
- Saves outputs (models, scaler/label encoder, metrics, plots, preds, tuning CSVs, evaluation report) to OUTPUT_FOLDER
- Uses TensorFlow / Keras + scikit-learn

Requirements:
    pip install numpy pandas scikit-learn matplotlib tensorflow joblib
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------
# Config - change if needed
# -----------------------
INPUT_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\18 neural networks\Neural networks\Alphabets_data.csv"
OUTPUT_FOLDER = r"D:\DATA SCIENCE\ASSIGNMENTS\18 neural networks\Neural networks"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Baseline model training settings
BASELINE_HIDDEN_UNITS = 128
BASELINE_DROPOUT = 0.2
BASELINE_LR = 1e-3
BASELINE_BATCH = 64
BASELINE_EPOCHS = 80
BASELINE_ES_PATIENCE = 8

# Tuning settings
RUN_TUNING = True         # Set False to skip random-search tuning
TUNING_N_ITER = 20        # Increase to 50+ for a more thorough search
TUNING_EPOCHS = 50
TUNING_PATIENCE = 7
SEED = 42

# -----------------------
# Utility functions
# -----------------------
def build_basic_ann(input_dim, num_classes, hidden_units=128, dropout_rate=0.2):
    model = Sequential([
        Dense(hidden_units, input_dim=input_dim, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax")
    ])
    return model

def build_model_with_hparams(input_dim, num_classes, num_layers, units, activation, dropout, learning_rate, optimizer_name):
    model = Sequential()
    model.add(Dense(units, activation=activation, input_dim=input_dim))
    if dropout and dropout > 0.0:
        model.add(Dropout(dropout))
    for _ in range(num_layers - 1):
        model.add(Dense(units, activation=activation))
        if dropout and dropout > 0.0:
            model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))

    if optimizer_name == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def plot_history(history, save_path):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("epoch"); plt.title("Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("epoch"); plt.title("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(cm, labels, csv_path, png_path, title="Confusion matrix"):
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(csv_path)
    plt.figure(figsize=(12,10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

def evaluate_model(model, X_test, y_test_onehot, label_encoder, out_prefix):
    """
    Evaluate a Keras model and write outputs:
      - classification_report text
      - confusion matrix CSV + PNG
      - predictions CSV (true_label, pred_label, pred_confidence)
    Returns a dict of aggregated metrics.
    """
    # Predict
    y_prob = model.predict(X_test)
    y_pred_int = np.argmax(y_prob, axis=1)
    y_true_int = np.argmax(y_test_onehot, axis=1)

    # Basic metrics
    acc = accuracy_score(y_true_int, y_pred_int)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_int, y_pred_int, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_int, y_pred_int, average="weighted", zero_division=0
    )

    # Per-class report
    cls_report = classification_report(y_true_int, y_pred_int, target_names=label_encoder.classes_, digits=4)

    # Save classification report
    with open(f"{out_prefix}_classification_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(cls_report)

    # Confusion matrix
    cm = confusion_matrix(y_true_int, y_pred_int)
    save_confusion_matrix(cm, label_encoder.classes_,
                         f"{out_prefix}_confusion_matrix.csv",
                         f"{out_prefix}_confusion_matrix.png",
                         title=f"{out_prefix} - Confusion matrix")

    # Save predictions
    pred_df = pd.DataFrame({
        "true_label": label_encoder.inverse_transform(y_true_int),
        "pred_label": label_encoder.inverse_transform(y_pred_int),
        "pred_confidence": np.max(y_prob, axis=1)
    })
    pred_df.to_csv(f"{out_prefix}_predictions.csv", index=False)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "n_test_samples": int(len(y_true_int))
    }
    # Save metrics JSON
    with open(f"{out_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

# -----------------------
# Main
# -----------------------
def main():
    # reproducibility (best effort; TF nondeterminism may still occur)
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load dataset
    print("Loading dataset from:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)
    if "letter" not in df.columns:
        raise ValueError("Target column 'letter' not found in the CSV.")
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Separate X, y
    X = df.drop(columns=["letter"])
    y = df["letter"]

    # Encode target
    le = LabelEncoder()
    y_int = le.fit_transform(y)                 # integers 0..N-1
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_int, num_classes=num_classes)
    print(f"Detected {num_classes} classes: {list(le.classes_)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=SEED, stratify=y_int
    )
    input_dim = X_train.shape[1]
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Save preprocessing objects
    joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "scaler.joblib"))
    joblib.dump(le, os.path.join(OUTPUT_FOLDER, "label_encoder.joblib"))

    # -----------------------
    # Baseline training
    # -----------------------
    print("\n=== Baseline model training ===")
    baseline_model = build_basic_ann(input_dim, num_classes, BASELINE_HIDDEN_UNITS, BASELINE_DROPOUT)
    baseline_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASELINE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    baseline_ckpt = os.path.join(OUTPUT_FOLDER, "baseline_best_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=BASELINE_ES_PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(baseline_ckpt, monitor="val_loss", save_best_only=True, verbose=1)
    ]
    history = baseline_model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=BASELINE_EPOCHS,
        batch_size=BASELINE_BATCH,
        callbacks=callbacks,
        verbose=2
    )
    baseline_final_path = os.path.join(OUTPUT_FOLDER, "baseline_final_model.h5")
    baseline_model.save(baseline_final_path)
    print(f"Baseline model saved: {baseline_final_path}")

    # Plot baseline history
    plot_history(history, os.path.join(OUTPUT_FOLDER, "baseline_training_history.png"))

    # Evaluate baseline on test set (and save artifacts)
    baseline_prefix = os.path.join(OUTPUT_FOLDER, "baseline")
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, le, baseline_prefix)
    print("\nBaseline metrics:", baseline_metrics)

    # -----------------------
    # Hyperparameter tuning (Random Search)
    # -----------------------
    best_tuned_path = os.path.join(OUTPUT_FOLDER, "best_tuned_model.h5")
    best_record = None

    if RUN_TUNING:
        print("\n=== Starting Random Search hyperparameter tuning ===")

        # Tuning search space
        search_space = {
            "num_layers": [1, 2, 3],
            "units": [32, 64, 128, 256],
            "activation": ["relu", "tanh", "elu"],
            "dropout": [0.0, 0.2, 0.4],
            "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            "batch_size": [32, 64, 128],
            "optimizer": ["adam", "sgd"]
        }

        def sample_config(space):
            return {k: random.choice(v) for k, v in space.items()}

        best_test_acc = -1.0
        results = []
        tuning_results_csv = os.path.join(OUTPUT_FOLDER, "hyperparam_tuning_results.csv")

        # For evaluation we need integer labels
        y_test_labels = np.argmax(y_test, axis=1)

        for i in range(1, TUNING_N_ITER + 1):
            cfg = sample_config(search_space)
            print(f"\n--- Iter {i}/{TUNING_N_ITER} | cfg: {cfg} ---")

            model = build_model_with_hparams(
                input_dim=input_dim,
                num_classes=num_classes,
                num_layers=cfg["num_layers"],
                units=cfg["units"],
                activation=cfg["activation"],
                dropout=cfg["dropout"],
                learning_rate=cfg["learning_rate"],
                optimizer_name=cfg["optimizer"]
            )

            iter_ckpt = os.path.join(OUTPUT_FOLDER, f"tmp_model_iter{i}.h5")
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=TUNING_PATIENCE, restore_best_weights=True, verbose=0),
                ModelCheckpoint(iter_ckpt, monitor="val_loss", save_best_only=True, verbose=0)
            ]

            t0 = time.time()
            history = model.fit(
                X_train, y_train,
                validation_split=0.15,
                epochs=TUNING_EPOCHS,
                batch_size=cfg["batch_size"],
                callbacks=callbacks,
                verbose=0
            )
            duration = time.time() - t0

            # Evaluate on test set
            y_pred_prob = model.predict(X_test)
            y_pred_int = np.argmax(y_pred_prob, axis=1)
            test_acc = accuracy_score(y_test_labels, y_pred_int)

            val_acc = max(history.history.get("val_accuracy", [0]))
            val_loss = min(history.history.get("val_loss", [float("inf")]))

            record = {
                "iter": i,
                "num_layers": cfg["num_layers"],
                "units": cfg["units"],
                "activation": cfg["activation"],
                "dropout": cfg["dropout"],
                "learning_rate": cfg["learning_rate"],
                "optimizer": cfg["optimizer"],
                "batch_size": cfg["batch_size"],
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "train_epochs_ran": len(history.history.get("loss", [])),
                "duration_sec": float(duration)
            }
            results.append(record)

            # Save per-iteration history
            hist_path = os.path.join(OUTPUT_FOLDER, f"history_iter{i}.json")
            with open(hist_path, "w") as hf:
                json.dump(history.history, hf)

            print(f"Iter {i} done | val_acc={val_acc:.4f} | test_acc={test_acc:.4f} | epochs={record['train_epochs_ran']} | {duration:.1f}s")

            # Save best model (by test acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_record = record
                model.save(best_tuned_path)
                plot_history(history, os.path.join(OUTPUT_FOLDER, f"best_history_iter{i}.png"))
                print(f"New best model saved to {best_tuned_path} (test_acc={test_acc:.4f})")

        # Save tuning summary CSV
        df_results = pd.DataFrame(results).sort_values("test_acc", ascending=False).reset_index(drop=True)
        df_results.to_csv(tuning_results_csv, index=False)
        print("\n=== Random search complete ===")
        print("Best record (by test_acc):")
        print(best_record)
        print("All tuning results saved to:", tuning_results_csv)

    else:
        print("\nRUN_TUNING is False: skipping hyperparameter search.")

    # -----------------------
    # Evaluation: compare baseline vs tuned (if available)
    # -----------------------
    evaluation_report_path = os.path.join(OUTPUT_FOLDER, "evaluation_report.txt")
    summary_csv_path = os.path.join(OUTPUT_FOLDER, "baseline_vs_tuned_summary.csv")
    summary_json_path = os.path.join(OUTPUT_FOLDER, "baseline_vs_tuned_summary.json")

    report_lines = []
    report_lines.append("Evaluation Report\n")
    report_lines.append(f"Dataset: {os.path.basename(INPUT_PATH)}")
    report_lines.append(f"Output folder: {OUTPUT_FOLDER}\n")
    report_lines.append("=== Baseline model metrics ===")
    for k, v in baseline_metrics.items():
        report_lines.append(f"{k}: {v}")
    report_lines.append("")

    tuned_metrics = None
    if RUN_TUNING and best_record is not None and os.path.exists(best_tuned_path):
        # load best tuned model and evaluate
        print("\nLoading best tuned model for evaluation:", best_tuned_path)
        tuned_model = load_model(best_tuned_path)
        tuned_prefix = os.path.join(OUTPUT_FOLDER, "tuned_best")
        tuned_metrics = evaluate_model(tuned_model, X_test, y_test, le, tuned_prefix)
        report_lines.append("=== Tuned model metrics (best) ===")
        for k, v in tuned_metrics.items():
            report_lines.append(f"{k}: {v}")
        report_lines.append("")
        report_lines.append("Best hyperparameter record (from tuning):")
        report_lines.append(json.dumps(best_record, indent=2))
        report_lines.append("")
    else:
        report_lines.append("No tuned model available (tuning skipped or no best model saved).")
        report_lines.append("")

    # Compare baseline vs tuned (if tuned exists)
    report_lines.append("=== Comparison summary ===")
    comparison_rows = []
    headers = ["metric", "baseline", "tuned", "delta (tuned - baseline)"]
    metrics_to_compare = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"]

    for metric_name in metrics_to_compare:
        base_val = baseline_metrics.get(metric_name, None)
        tuned_val = tuned_metrics.get(metric_name, None) if tuned_metrics else None
        delta = (tuned_val - base_val) if (base_val is not None and tuned_val is not None) else None
        comparison_rows.append({
            "metric": metric_name,
            "baseline": base_val,
            "tuned": tuned_val,
            "delta": delta
        })
        line = f"{metric_name}: baseline={base_val} | tuned={tuned_val} | delta={delta}"
        report_lines.append(line)

    # Short discussion (automated template + pointers)
    report_lines.append("\nDiscussion / Observations:")
    if tuned_metrics:
        acc_delta = tuned_metrics["accuracy"] - baseline_metrics["accuracy"]
        report_lines.append(f"- Overall test accuracy changed by {acc_delta:.6f} (tuned - baseline).")
        report_lines.append("- If tuned model shows improvement, likely causes include better learning rate / depth / regularization choices from random search.")
        report_lines.append("- If tuned model did not improve, possible reasons:")
        report_lines.append("  * search space did not cover the region with better hyperparameters")
        report_lines.append("  * insufficient search budget (increase TUNING_N_ITER)")
        report_lines.append("  * model architecture capacity / dataset size mismatch")
        report_lines.append("- Recommendations:")
        report_lines.append("  * run a focused local grid around the best lr/units found")
        report_lines.append("  * try Keras Tuner (Hyperband or Bayesian) for smarter sampling")
        report_lines.append("  * consider data augmentation, feature engineering, or deeper architectures if underfitting")
    else:
        report_lines.append("- No tuned model to compare; baseline metrics reported above.")
        report_lines.append("- To get a tuned model: set RUN_TUNING = True and increase TUNING_N_ITER.")

    # Save evaluation report
    with open(evaluation_report_path, "w") as f:
        f.write("\n".join(report_lines))

    # Save CSV and JSON comparison
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w") as f:
        json.dump({"baseline_metrics": baseline_metrics, "tuned_metrics": tuned_metrics, "comparison": comparison_rows, "best_record": best_record}, f, indent=2)

    print("\nEvaluation complete.")
    print("Saved evaluation report to:", evaluation_report_path)
    print("Saved comparison CSV to:", summary_csv_path)
    print("Saved summary JSON to:", summary_json_path)

if __name__ == "__main__":
    main()
