# titanic_eda_xgbm_lgbm.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === PATH SETUP ===
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\15 XGBM & LGBM\XGBM & LGBM"
train_path = os.path.join(base_path, "Titanic_train.csv")
test_path = os.path.join(base_path, "Titanic_test.csv")

# === LOAD DATA ===
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("✅ Datasets Loaded Successfully")
print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
print("\n--- Columns ---")
print(list(train_df.columns))

# === 1️⃣ MISSING VALUES ===
print("\n--- Missing Values in Train Dataset ---")
print(train_df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.heatmap(train_df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap - Train Data")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "missing_values_heatmap.png"))
plt.close()

# === 2️⃣ FEATURE DISTRIBUTIONS ===
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(12, 8))
train_df[numeric_cols].hist(bins=15, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions - Titanic Dataset", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "histograms.png"))
plt.close()

# Boxplots to see outliers
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=train_df[col], color='salmon')
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "boxplots.png"))
plt.close()

# === 3️⃣ RELATIONSHIPS WITH SURVIVAL ===

# Bar plot: Survival vs Sex
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Sex', hue='Survived', palette='pastel')
plt.title("Survival Count by Sex")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "survival_by_sex.png"))
plt.close()

# Bar plot: Survival vs Pclass
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Pclass', hue='Survived', palette='muted')
plt.title("Survival Count by Passenger Class")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "survival_by_pclass.png"))
plt.close()

# Scatter: Age vs Fare colored by survival
plt.figure(figsize=(7, 5))
sns.scatterplot(data=train_df, x='Age', y='Fare', hue='Survived', palette='coolwarm', alpha=0.7)
plt.title("Age vs Fare — Colored by Survival")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "scatter_age_fare.png"))
plt.close()

# Boxplot: Age distribution by Survival
plt.figure(figsize=(6, 4))
sns.boxplot(data=train_df, x='Survived', y='Age', palette='Set2')
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "age_by_survival.png"))
plt.close()

# Bar plot: Survival vs Embarked
if 'Embarked' in train_df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train_df, x='Embarked', hue='Survived', palette='pastel')
    plt.title("Survival by Embarked Port")
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "survival_by_embarked.png"))
    plt.close()

print("\n✅ EDA Completed Successfully.")
print("Plots saved in:", base_path)
print("""
Saved files:
- missing_values_heatmap.png
- histograms.png
- boxplots.png
- survival_by_sex.png
- survival_by_pclass.png
- scatter_age_fare.png
- age_by_survival.png
- survival_by_embarked.png
""")

# === QUICK INSIGHTS ===
print("\n--- Insights ---")
print("1. Females had a higher survival rate compared to males.")
print("2. Higher Passenger Classes (1st class) show higher survival chances.")
print("3. Younger passengers and those paying higher fares tended to survive more.")
print("4. Age and Fare contain outliers but are still informative.")
print("5. Missing values primarily in 'Age', 'Cabin', and 'Embarked' columns.")
