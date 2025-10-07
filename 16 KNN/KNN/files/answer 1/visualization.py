import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\16 KNN\KNN\Zoo.csv"
zoo_df = pd.read_csv(file_path)

# Drop non-numeric column
zoo_df = zoo_df.drop(columns=["animal name"])

# 1. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(zoo_df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 2. Pairplot (simplified to avoid overload)
selected_features = ['hair', 'feathers', 'eggs', 'milk', 'aquatic', 'legs', 'type']
sns.pairplot(zoo_df[selected_features], hue='type', diag_kind="hist", palette='viridis')
plt.suptitle("Pairwise Relationships Between Key Features", y=1.02)
plt.show()

# 3. Distribution of 'legs'
plt.figure(figsize=(8, 5))
sns.countplot(x='legs', data=zoo_df, palette='magma')
plt.title('Distribution of Number of Legs')
plt.xlabel('Legs')
plt.ylabel('Count')
plt.show()

# 4. Presence of hair across animal types
plt.figure(figsize=(8, 5))
sns.countplot(x='type', hue='hair', data=zoo_df, palette='Set2')
plt.title('Presence of Hair Across Animal Types')
plt.xlabel('Animal Type')
plt.ylabel('Count')
plt.legend(title='Hair')
plt.show()
