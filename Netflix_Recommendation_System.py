#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

class MovieRecommender:
    def __init__(self):
        self.recommended_movies = []

    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Further preprocessing steps (e.g., remove punctuation, stopwords) can be added here
        return text

    def get_recommendations_clustering(self, movie_title, netflix_data, netflix, num_recommendations=5):
        # Find the cluster of the given movie
        movie_cluster = netflix_data.loc[netflix_data['title'].str.lower() == movie_title.lower(), 'kcluster'].values[0]

        # Filter movies from the same cluster
        recommended_movies = netflix[netflix['kcluster'] == movie_cluster]

        # Exclude the movie the user already watched
        recommended_movies = recommended_movies[recommended_movies['title'].str.lower() != movie_title.lower()]

        # Calculate centroid of the cluster
        cluster_centroid = netflix_data.loc[netflix_data['kcluster'] == movie_cluster, netflix_data.columns != 'kcluster'].mean()

        # Calculate distance of each recommended movie from the cluster centroid
        recommended_movies['distance'] = recommended_movies.apply(lambda row: np.linalg.norm(row.drop(['title', 'kcluster']) - cluster_centroid), axis=1)

        # Select the top 5 movies with the least distance
        top_recommendations = recommended_movies.sort_values(by='distance').head(num_recommendations)

        return top_recommendations

    def get_recommendations_tfidf(self, user_input, num_recommendations=5):
        # Sample dataset (Replace this with your actual dataset loading code)
        df = pd.read_csv(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 3\EDA\netflix_Clusters.csv', encoding='latin1')

        # Preprocess the text
        df['cleaned_description'] = df['description'].apply(self.preprocess_text)

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['cleaned_description'])

        # Convert user input to TF-IDF vector
        user_tfidf_vector = vectorizer.transform([self.preprocess_text(user_input)])

        # Calculate cosine similarity
        cos_similarities = cosine_similarity(user_tfidf_vector, tfidf_matrix)

        # Exclude already recommended movies
        for idx, _ in enumerate(cos_similarities):
            if df.loc[idx, 'title'] in self.recommended_movies:
                cos_similarities[0][idx] = 0

        # Get indices of the descriptions with highest similarity
        best_match_indices = cos_similarities.argsort()[0][-num_recommendations:][::-1]

        recommendations = []
        for idx in best_match_indices:
            # Retrieve corresponding movie title and description
            movie_title = df.loc[idx, 'title']
            movie_description = df.loc[idx, 'description']
            movie_image = self.get_movie_image(movie_title)  # Fetch movie image
            recommendations.append((movie_title, movie_description, movie_image))
            self.recommended_movies.append(movie_title)

        return recommendations

    def get_movie_image(self, title):
        api_key = '41718a96'  # Replace with your actual OMDB API key
        url = f"http://www.omdbapi.com/?apikey={api_key}&t={title}"
        response = requests.get(url)
        data = response.json()
        poster_url = data.get('Poster', None)
        if poster_url and poster_url != 'N/A':
            return poster_url
        else:
            return None

    def get_movie_details(self, title):
        api_key = '41718a96'  # Replace with your actual OMDB API key
        url = f"http://www.omdbapi.com/?apikey={api_key}&t={title}"
        response = requests.get(url)
        data = response.json()
        return {
            'Title': data.get('Title', ''),
            'Poster': data.get('Poster', ''),
            'Description': data.get('Plot', '')
        }

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Movie Recommendations (Based on Clustering)", "Movie Recommendations (Based on Similarity)"))

    movie_recommender = MovieRecommender()  # Initialize MovieRecommender object

    if page == "Movie Recommendations (Based on Clustering)":
        recommend_movies_based_on_clustering(movie_recommender)  # Pass movie_recommender object
    elif page == "Movie Recommendations (Based on Similarity)":
        recommend_movies_based_on_tfidf(movie_recommender)  # Pass movie_recommender object

def recommend_movies_based_on_clustering(movie_recommender):  
    with open(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 3\Final\kprototypes_model.pkl', 'rb') as f:
        kproto = pickle.load(f)

    with open(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 3\Final\clustered_data.pkl', 'rb') as f:
        netflix_data = pickle.load(f)

    netflix = pd.read_csv(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 3\EDA\netflix_Model.csv', encoding='latin1')

    st.title("Movie Recommendation App")
    st.sidebar.title("User Input")

    movie_title = st.sidebar.text_input("Enter a movie title:")

    if st.sidebar.button("Get Recommendations"):
        recommendations = movie_recommender.get_recommendations_clustering(movie_title, netflix_data, netflix)
        if not recommendations.empty:
            st.write("Top 5 recommended movies:")
            for i, (_, recommendation) in enumerate(recommendations.iterrows(), 1):
                st.write(f"{i}. {recommendation['title']}")
                movie_details = movie_recommender.get_movie_details(recommendation['title'])
                st.write(f"Description: {movie_details['Description']}")
                poster_url = movie_recommender.get_movie_image(recommendation['title'])
                if poster_url:
                    st.image(poster_url, caption=recommendation['title'], use_column_width=True)
                else:
                    st.write("Movie image not found.")
                st.write("---")
        else:
            st.warning("No recommendations found.")

def recommend_movies_based_on_tfidf(movie_recommender): 
    st.title("Movie Recommendation System")
    user_input = st.text_input("Enter your query:")
    if st.button("Get Recommendations"):
        if user_input:
            recommendations = movie_recommender.get_recommendations_tfidf(user_input)
            if recommendations:
                st.write("Top 5 recommended movies:")
                for i, (movie_title, movie_description, movie_image) in enumerate(recommendations, 1):
                    st.write(f"{i}. {movie_title}")
                    st.write("Description:", movie_description)
                    if movie_image:
                        st.image(movie_image, caption=movie_title, use_column_width=True)
                    else:
                        st.write("Image not found.")
                    st.write("---")
            else:
                st.warning("No recommendations found.")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




