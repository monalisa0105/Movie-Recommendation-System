# 🎬 Movie Recommendation System

A **content-based movie recommendation system** that suggests movies similar to a user-selected title by analyzing movie metadata such as genres and overview descriptions. This project uses natural language processing techniques like TF-IDF vectorization and cosine similarity to find and recommend the top 5 most similar movies.

## 🧠 Concepts Used

- **Content-Based Filtering:** Recommending movies similar to a selected movie based on their attributes (genres, overview).
- **Text Preprocessing:** Cleaning and combining movie metadata for analysis.
- **TF-IDF Vectorization:** Converting text into numerical vectors that reflect the importance of words.
- **Cosine Similarity:** Measuring the similarity between movie vectors to find close matches.
- **Natural Language Processing (NLP):** Extracting meaningful features from movie descriptions.


## ✨ Features

- Recommends movies based on content similarity rather than user ratings.
- Uses both genres and movie overview text for more accurate recommendations.
- Handles missing data gracefully.
- Provides top 5 similar movie recommendations.
- Interactive dropdown for movie selection.
- Built with Streamlit for easy web deployment.

## 🌐 Deployment

Check out the live app here:  
[https://monalisa0105-movie-recommendation-system-app-ke9ign.streamlit.app/](https://monalisa0105-movie-recommendation-system-app-ke9ign.streamlit.app/)


## 📊 Dataset

The recommendations are powered by the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle.

