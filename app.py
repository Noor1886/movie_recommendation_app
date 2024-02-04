import streamlit as st
import pandas as pd
import pickle
# import altair as alt
from movie_app import get_top5_recommendations
from movie_app import genre_features


# Load model and data
with open("/home/abdul/movie_recommendation_app/movie_app.pkl", "rb") as f:
    model = pickle.load(f)

movies_df = pd.read_csv("/home/abdul/movie_recommendation_app/movies.csv")
# genre_features = pd.read_csv("/home/abdul/movie_recommendation_app/genre_features.csv")  # Load genre features separately

# Function to get recommendations for a given user ID
def get_recommendations(user_id):
    movie_ids = movies_df['movieId'].values
    top5_movie_ids, top5_ratings = get_top5_recommendations(model, user_id, movie_ids, genre_features)
    recommended_movies = movies_df[movies_df['movieId'].isin(top5_movie_ids)]
    return recommended_movies, top5_ratings

# Homepage
st.title("Movie Recommendation App")
st.write("Welcome to the movie recommendation app! Discover movies you'll love based on your preferences.")

user_id = st.number_input("Enter your user ID:", min_value=1)

if user_id:
    # Recommendations Page
    recommended_movies, top5_ratings = get_recommendations(user_id)
    st.write("Here are your top 5 movie recommendations:")
    st.dataframe(recommended_movies[['title', 'genres']].assign(Predicted_Rating=top5_ratings))

    # Add option to view movie details
    selected_movie_id = st.selectbox("Select a movie to view details:", recommended_movies['movieId'])
    if selected_movie_id:
        selected_movie = recommended_movies[recommended_movies['movieId'] == selected_movie_id]
        st.write(selected_movie)

# About Page
st.sidebar.title("About")
st.sidebar.write("This app uses a collaborative filtering model trained on movie ratings and genres to predict user preferences.")
st.sidebar.write("Deployment Platform: Streamlit Community Cloud")
