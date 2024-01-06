import pandas as pd
from sqlalchemy import create_engine
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy

# Database connection
db_url = "postgresql://WieteskaKarolina:Ur3LV9DJPdAB@ep-shrill-union-593741.eu-central-1.aws.neon.tech/movie_app_db"
engine = create_engine(db_url)

# SQL queries to fetch user ratings and watch_later data
query_rating = "SELECT user_id, movie_id, rating FROM public.rating"
query_watch_later = "SELECT user_id, movie_id FROM public.watch_later"

# Fetch data from the database
with engine.connect() as connection:
    result_rating = connection.execute(query_rating)
    result_watch_later = connection.execute(query_watch_later)
    df_rating = pd.DataFrame(result_rating.fetchall(), columns=result_rating.keys())
    df_watch_later = pd.DataFrame(result_watch_later.fetchall(), columns=result_watch_later.keys())

# Merge rating and watch_later DataFrames
df_combined = pd.concat([df_rating, df_watch_later], ignore_index=True)

# Drop rows with missing ratings
df_combined = df_combined.dropna(subset=['rating'])

# Create a Surprise reader
reader = Reader(rating_scale=(1, 5))

# Load the data into a Surprise dataset
data = Dataset.load_from_df(df_combined[['user_id', 'movie_id', 'rating']], reader)

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use a basic collaborative filtering algorithm (KNN)
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)

# Train the algorithm on the trainset
algo.fit(trainset)

# Make predictions on the testset
predictions = algo.test(testset)

# Evaluate the accuracy of the model (optional)
accuracy.rmse(predictions)

# Get recommendations for a specific user
user_id_to_recommend = 1

# Get movies to predict, excluding those in the watch_later list
movies_to_predict = df_combined[df_combined['user_id'] == user_id_to_recommend]['movie_id'].unique()

# Get predicted ratings for the movies
movie_ratings = [(movie_id, algo.predict(user_id_to_recommend, movie_id).est) for movie_id in movies_to_predict]

# Sort movies by predicted ratings
recommended_movies = sorted(movie_ratings, key=lambda x: x[1], reverse=True)[:10]

print(recommended_movies)
