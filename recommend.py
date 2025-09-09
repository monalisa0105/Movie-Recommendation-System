import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

df = pd.read_csv('tmdb_5000_movies.csv')

def extract_genres(genres_str):
    genres_list = ast.literal_eval(genres_str)
    return " ".join([genre['name'] for genre in genres_list])

df['genres'] = df['genres'].apply(extract_genres)
df['overview'] = df['overview'].fillna('')
df['text_features'] = df['genres'] + " " + df['overview']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend_movies(title, cosine_sim=cosine_sim):
    if title not in indices:
        return "Movie not found in dataset."

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

movie_name = "The Dark Knight"
recommended = recommend_movies(movie_name)

print(f"\nTop 5 movies similar to '{movie_name}':")
for movie in recommended:
    print("➤", movie)
