import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AnimeRecommender:
    def __init__(self, anime_csv_path='anime.csv'):
        self.anime = pd.read_csv(anime_csv_path)
        self.anime['genre'] = self.anime['genre'].fillna('')
        self._prepare_model()

    def _prepare_model(self):
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.anime['genre'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.anime.index, index=self.anime['name']).drop_duplicates()

    def recommend(self, title, top_n=5):
        if title not in self.indices:
            return f"Anime titled '{title}' not found in the dataset."
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        anime_indices = [i[0] for i in sim_scores]
        return self.anime['name'].iloc[anime_indices].tolist()

# Example usage:
# rec = AnimeRecommender()
# print(rec.recommend('Anime 1'))
