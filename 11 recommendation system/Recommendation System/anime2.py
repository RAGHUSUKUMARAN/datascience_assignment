import pandas as pd

# Load dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\11 recommendation system\Recommendation System\anime.csv"
anime_df = pd.read_csv(file_path)

# Convert "episodes" to numeric and handle missing values
anime_df['episodes'] = pd.to_numeric(anime_df['episodes'].replace("Unknown", None), errors='coerce')
anime_df['genre'] = anime_df['genre'].fillna("Unknown")
anime_df['type'] = anime_df['type'].fillna("Unknown")
anime_df['rating'] = anime_df['rating'].fillna(anime_df['rating'].mean())
