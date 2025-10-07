import pandas as pd

# Load the dataset
file_path = '/mnt/data/anime.csv'
anime_df = pd.read_csv(file_path)

# Display first few rows and dataset info
anime_df.head(), anime_df.info()
