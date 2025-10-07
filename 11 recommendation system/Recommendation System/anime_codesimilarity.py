"""
anime_recommender.py
Drop this file in the same environment where you have anime_features.csv
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from typing import Union, Optional

# --------- CONFIG: update if your path differs ----------
FEATURES_PATH = r"D:\DATA SCIENCE\ASSIGNMENTS\11 recommendation system\anime_features.csv"
# -------------------------------------------------------

class AnimeRecommender:
    def __init__(self, features_path: str = FEATURES_PATH, compute_matrix: bool = True):
        # Load precomputed features
        self.df = pd.read_csv(features_path)
        # Identify feature columns (everything except anime_id & name)
        self.id_col = 'anime_id'
        self.name_col = 'name'
        self.feature_cols = [c for c in self.df.columns if c not in {self.id_col, self.name_col}]
        # Feature matrix
        self.features = self.df[self.feature_cols].values.astype(float)
        # Precompute cosine similarity matrix if requested
        self.sim_matrix = None
        if compute_matrix:
            self.sim_matrix = cosine_similarity(self.features)
            # Keep numerical stability: clip tiny negatives to 0, and 1 on diagonal
            np.fill_diagonal(self.sim_matrix, 1.0)

    def _get_index_for_title(self, title: Union[int, str]) -> Optional[int]:
        """
        Return dataframe index for a given anime name or anime_id.
        If string given, tries exact match (case-insensitive), then fuzzy match.
        """
        if isinstance(title, int) or (isinstance(title, str) and title.isdigit()):
            # try anime_id
            try:
                anime_id = int(title)
                matches = self.df[self.df[self.id_col] == anime_id]
                if not matches.empty:
                    return matches.index[0]
            except ValueError:
                pass

        if isinstance(title, str):
            # exact case-insensitive match
            mask = self.df[self.name_col].str.lower() == title.lower()
            if mask.any():
                return mask.idxmax()

            # partial substring match
            substr_mask = self.df[self.name_col].str.lower().str.contains(title.lower())
            if substr_mask.any():
                return substr_mask[substr_mask].index[0]

            # fuzzy match fallback
            choices = self.df[self.name_col].tolist()
            close = get_close_matches(title, choices, n=1, cutoff=0.6)
            if close:
                return self.df[self.name_col] == close[0].__str__() and self.df[self.name_col][self.df[self.name_col] == close[0]].index[0]
        return None

    def recommend_anime(self,
                        target: Union[str, int],
                        top_n: int = 10,
                        threshold: Optional[float] = None,
                        include_target: bool = False) -> pd.DataFrame:
        """
        Recommend similar anime for a given target (title string or anime_id).
        Parameters:
            - target: anime title (str) or anime_id (int or numeric string)
            - top_n: return up to top_n recommendations (ignored if threshold used and fewer results)
            - threshold: float in [0,1]. If provided, returns all anime with similarity >= threshold.
                         If None, returns top_n highest-similarity anime.
            - include_target: whether to include the target anime itself in results (default False)
        Returns:
            pandas DataFrame with columns: anime_id, name, similarity
        """
        idx = self._get_index_for_title(target)
        if idx is None:
            raise ValueError(f"Target '{target}' not found (no close matches). Check spelling or use anime_id.")

        # compute similarity row if matrix isn't precomputed
        if self.sim_matrix is not None:
            sims = self.sim_matrix[idx]
        else:
            sims = cosine_similarity(self.features[idx:idx+1], self.features).flatten()

        results = pd.DataFrame({
            self.id_col: self.df[self.id_col],
            self.name_col: self.df[self.name_col],
            'similarity': sims
        })

        # Optionally drop target
        if not include_target:
            results = results[results[self.id_col] != self.df.loc[idx, self.id_col]]

        # Apply threshold or top_n
        if threshold is not None:
            filtered = results[results['similarity'] >= float(threshold)].sort_values('similarity', ascending=False)
            return filtered.reset_index(drop=True)
        else:
            top = results.sort_values('similarity', ascending=False).head(top_n).reset_index(drop=True)
            return top

# ---------------- Example usage ----------------
if __name__ == "__main__":
    rec = AnimeRecommender(FEATURES_PATH)
    # Example 1: by exact title
    print("Top 8 similar to 'Fullmetal Alchemist: Brotherhood':")
    print(rec.recommend_anime("Fullmetal Alchemist: Brotherhood", top_n=8))

    # Example 2: fuzzy / partial ti
