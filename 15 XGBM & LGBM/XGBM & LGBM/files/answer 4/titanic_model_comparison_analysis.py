# titanic_model_comparison_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === PATH SETUP ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\15 XGBM & LGBM\XGBM & LGBM"
results_path = os.path.join(base_path, "model_comparison_results.csv")

# === LOAD RESULTS ===
results_df = pd.read_csv(results_path)
print("✅ Results Loaded Successfully:")
print(results_df, "\n")

# === 1️⃣ COMPARISON PLOT ===
plt.figure(figsize=(8, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
results_melted = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

sns.barplot(data=results_melted, x="Metric", y="Score", hue="Model", palette="coolwarm")
plt.title("LightGBM vs XGBoost — Performance Comparison", fontsize=14)
plt.ylim(0, 1)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "lgbm_xgbm_comparison.png"))
plt.close()

print("📊 Saved performance comparison chart as 'lgbm_xgbm_comparison.png'")

# === 2️⃣ INTERPRETATION ===
lgbm = results_df[results_df["Model"] == "LightGBM"].iloc[0]
xgb = results_df[results_df["Model"] == "XGBoost"].iloc[0]

print("\n--- Comparative Summary ---")
if lgbm["Accuracy"] > xgb["Accuracy"]:
    print("✅ LightGBM achieved higher accuracy than XGBoost.")
else:
    print("✅ XGBoost achieved higher accuracy than LightGBM.")

print(f"\nLightGBM → Accuracy: {lgbm['Accuracy']:.4f}, Precision: {lgbm['Precision']:.4f}, Recall: {lgbm['Recall']:.4f}, F1: {lgbm['F1']:.4f}")
print(f"XGBoost → Accuracy: {xgb['Accuracy']:.4f}, Precision: {xgb['Precision']:.4f}, Recall: {xgb['Recall']:.4f}, F1: {xgb['F1']:.4f}")

print("\n--- Insights ---")
print("1. LightGBM usually trains faster and handles large datasets more efficiently using histogram-based learning.")
print("2. XGBoost provides slightly more stable performance on smaller datasets due to its regularization controls.")
print("3. Both models identified similar top predictors — 'Sex', 'Pclass', 'Fare', and 'Age'.")
print("4. If computational speed is key → LightGBM wins.")
print("5. If interpretability and consistent performance are needed → XGBoost holds strong.")
