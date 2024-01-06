import os
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor

# Set the Hugging Face token
os.environ["HF_HOME"] = "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"

# Download and load the sentiment analysis pipeline locally
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
except Exception as e:
    print(f"Error loading sentiment analysis pipeline: {e}")
    sentiment_pipeline = None


def get_sentiment_score(text, max_length=512):
    if sentiment_pipeline is None:
        return None

    # Truncate the text to fit within the specified maximum length
    truncated_text = text[:max_length]

    # Perform sentiment analysis on the truncated input
    result = sentiment_pipeline(truncated_text)
    return result[0]['score']


# Database connection and query
db_url = "postgresql://WieteskaKarolina:Ur3LV9DJPdAB@ep-shrill-union-593741.eu-central-1.aws.neon.tech/movie_app_db"
engine = create_engine(db_url)
query = "SELECT user_id, movie_id, text, likes FROM public.comment"

# Explicitly open and close a connection
with engine.connect() as connection:
    result = connection.execute(text(query))
    df_sql_comment = pd.DataFrame(result.fetchall(), columns=result.keys())

# Apply sentiment analysis to the comments
df_sql_comment['sentiment_score'] = df_sql_comment['text'].apply(get_sentiment_score)

# Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_sql_comment['text'])

# Calculate cosine similarity between comments
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Function to get top N similar comments for a given comment index
def get_top_similar_comments(comment_index, N=5):
    sim_scores = list(enumerate(cosine_sim[comment_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar_indices = [i for i, _ in sim_scores[1:N + 1]]
    return df_sql_comment['text'].iloc[top_similar_indices]


def parallel_get_top_similar_comments(comment_indices, N=5):
    with ThreadPoolExecutor() as executor:
        top_similar_comments_list = list(executor.map(lambda idx: get_top_similar_comments(idx, N), comment_indices))
    return top_similar_comments_list


def recommend_movies_for_user(user_id, N=5):
    user_comments = df_sql_comment[df_sql_comment['user_id'] == user_id]

    if user_comments['sentiment_score'].isna().all():
        print(f"No sentiment scores available for user {user_id}. Unable to make recommendations.")
        return

    # Calculate the weighted average sentiment score based on likes
    weighted_sentiment = (user_comments['sentiment_score'] * user_comments['likes']).sum() / user_comments[
        'likes'].sum()

    # Find the comment with the highest sentiment score in the user's comments
    max_sentiment_comment_index = user_comments['sentiment_score'].idxmax()

    # Get top N similar comments for the comment with the highest sentiment score using parallel processing
    comment_indices = range(len(df_sql_comment))
    top_similar_comments_list = parallel_get_top_similar_comments(comment_indices, N)
    top_similar_comments = pd.concat(top_similar_comments_list)

    # Print the recommendations (you might want to return the result instead)
    print(f"Weighted Sentiment Score for user {user_id}: {weighted_sentiment}")
    print(f"Top {N} similar comments:")
    print(top_similar_comments)

    # Additional step: Recommend movies based on sentiment and likes
    recommended_movies = user_comments.sort_values(by='sentiment_score', ascending=False).head(N)

    # Print recommended movies (you might want to return the result instead)
    print(f"Recommended Movies for user {user_id}:")
    print(recommended_movies[['movie_id', 'text', 'likes', 'sentiment_score']])


# Example: Recommend movies for user with ID 8
recommend_movies_for_user(user_id=8, N=5)
