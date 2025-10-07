import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\16 KNN\KNN\Zoo.csv"
zoo_df = pd.read_csv(file_path)

# Display first few rows, structure, and basic info
zoo_head = zoo_df.head()
zoo_info = zoo_df.info()
zoo_description = zoo_df.describe()

# Class distribution plot
plt.figure(figsize=(8, 5))
sns.countplot(x='type', data=zoo_df, palette='viridis')
plt.title('Distribution of Animal Types')
plt.xlabel('Animal Type')
plt.ylabel('Count')
plt.show()

zoo_head, zoo_description
