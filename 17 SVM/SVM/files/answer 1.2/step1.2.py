# Step 2: Visualizing Feature Distributions

import matplotlib.pyplot as plt
import seaborn as sns

# --- Categorical Features ---
cat_features = ['cap_shape', 'cap_surface', 'cap_color', 'odor', 'habitat', 'class']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(cat_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, data=mushroom_df, palette='viridis')
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Numerical Features ---
num_features = ['stalk_height', 'cap_diameter']

# Histograms
plt.figure(figsize=(12, 5))
for i, feature in enumerate(num_features, 1):
    plt.subplot(1, 2, i)
    sns.histplot(mushroom_df[feature], kde=True, color='teal')
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(12, 5))
for i, feature in enumerate(num_features, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(x=mushroom_df[feature], color='orange')
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()
