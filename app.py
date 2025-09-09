import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

@st.cache_data
def load_data():
    df = pd.read_csv('tmdb_5000_movies.csv')
    return df

@st.cache_data
def preprocess_data(df):
    def extract_genres(genres_str):
        genres_list = ast.literal_eval(genres_str)
        return " ".join([genre['name'] for genre in genres_list])

    df['genres'] = df['genres'].apply(extract_genres)
    df['overview'] = df['overview'].fillna('')
    df['text_features'] = df['genres'] + " " + df['overview']
    return df

@st.cache_data
def create_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, indices

def recommend_movies(title, cosine_sim, indices, df):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 excluding itself
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

def main():
    st.title("Movie Recommendation System 🎬")
    st.write("Select a movie to get recommendations:")

    df = load_data()
    df = preprocess_data(df)
    cosine_sim, indices = create_similarity_matrix(df)

    # Dropdown menu for selecting movie title
    movie_name = st.selectbox("Choose a movie:", options=df['title'].sort_values().unique())

    if movie_name:
        recommended = recommend_movies(movie_name, cosine_sim, indices, df)
        if recommended:
            st.subheader(f"Top 5 movies similar to '{movie_name}':")
            for movie in recommended:
                st.write(f"➤ {movie}")
        else:
            st.error("Movie not found in dataset. Please try another title.")

if __name__ == "__main__":
    main()
