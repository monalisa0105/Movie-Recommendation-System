# 🎬 Movie Recommendation System | Content-Based Filtering

Welcome to a simple yet powerful **Movie Recommendation System** that suggests movies similar to your favorites! Built using Python, this project uses **content-based filtering** to analyze genres and descriptions and recommends top 5 similar movies using **TF-IDF** and **cosine similarity**.

## 🚀 Project Overview

This project demonstrates how to build a **Content-Based Movie Recommender** using metadata from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).  
It works by comparing the **genre** and **overview** of movies to find the most similar titles.

## ✨ Features

- Content-based movie recommendations  
- Uses genres and descriptions  
- TF-IDF vectorization for text data  
- Cosine similarity for finding similar movies  
- Handles missing data  
- Returns top 5 recommendations  
- Simple movie title input  
- Easily extendable  

## 🧠 What You’ll get to see

- 📌 Text Preprocessing
- 📌 TF-IDF Vectorization
- 📌 Cosine Similarity
- 📌 Content-Based Filtering Techniques
- 📌 Movie Recommendation Logic

## 🗃️ Files in This Repository

| File Name              | Description                                  |
|------------------------|----------------------------------------------|
| `recommend.py`         | Main Python script to run the recommender    |
| `tmdb_5000_movies.csv` | Movie metadata dataset (from Kaggle)         |
| `README.md`            | This file – complete project documentation   |

## 📊 Dataset Used

- **Name**: TMDB 5000 Movie Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)  
- **File Used**: `tmdb_5000_movies.csv`

---

## ⚙️ How to Run This Project

1. ✅ **Install Required Libraries**

```bash
pip install pandas scikit-learn
