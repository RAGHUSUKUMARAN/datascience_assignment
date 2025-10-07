import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# Load dataset from your local path
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\11 recommendation system\Recommendation System\anime.csv"
anime = pd.read_csv(file_path)

# --- Step 1 Cleaning ---
anime['episodes'] = pd.to_numeric(anime['episodes'].replace("Unknown", None), errors='coerce')
anime['genre'] = anime['genre'].fillna("Unknown")
anime['type'] = anime['type'].fillna("Unknown")
anime['rating'] = anime['rating'].fillna(anime['rating'].mean())

# --- Step 2 Feature Extraction ---
def split_genres(genres):
    if not genres or genres == "Unknown":
        return []
    return [g.strip() for g in genres.split(',') if g.strip()]

anime['genre_list'] = anime['genre'].apply(split_genres)

# Multi-hot encode genres
mlb = MultiLabelBinarizer(sparse_output=False)
genre_matrix = mlb.fit_transform(anime['genre_list'])
genre_cols = [f"genre__{g}" for g in mlb.classes_]
genre_df = pd.DataFrame(genre_matrix, columns=genre_cols, index=anime.index)

# Normalize rating and members
scaler = MinMaxScaler()
num_df = anime[['rating', 'members']].copy()
num_df[['rating_norm', 'members_norm']] = scaler.fit_transform(num_df[['rating', 'members']])

# Combine all features
features_df = pd.concat([anime[['anime_id', 'name']], genre_df, num_df[['rating_norm', 'members_norm']]], axis=1)

# --- Save processed file to your folder ---
out_path = r"D:\DATA SCIENCE\ASSIGNMENTS\11 recommendation system\anime_features.csv"
features_df.to_csv(out_path, index=False)

print("✅ Feature extraction complete!")
print(f"Shape of feature matrix: {features_df.shape}")
print(f"File saved to: {out_path}")
