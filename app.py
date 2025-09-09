import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    return pd.read_csv('tmdb_5000_movies.csv')

df = load_data()

def combine_features(row):
    genres = row['genres'] if isinstance(row['genres'], str) else ''
    overview = row['overview'] if isinstance(row['overview'], str) else ''
    return genres + ' ' + overview

df['combined'] = df.apply(combine_features, axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(title):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices.get(title)

    if idx is None:
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 excluding itself

    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

st.title("🎬 Movie Recommendation System")

movie_input = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if movie_input:
        recommendations = get_recommendations(movie_input)
        if recommendations:
            st.write(f"Top 5 movies similar to '{movie_input}':")
            for movie in recommendations:
                st.write(f"➤ {movie}")
        else:
            st.write("Movie not found. Please try another title.")
    else:
        st.write("Please enter a movie title.")
