from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost:5432/yourdatabase'
db = SQLAlchemy(app)


# Assuming you have a UserRating model
class UserRating(db.Model):
    __tablename__ = 'user_ratings'
    user_id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer)


# Load data into Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(UserRating[['user_id', 'movie_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Build and train the collaborative filtering model
model = SVD()
model.fit(trainset)


# API endpoint to get movie recommendations for a user
@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    # Assuming you have a function to get unrated movies for a user
    unrated_movies = get_unrated_movies(user_id)

    # Predict ratings for unrated movies
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movies]

    # Get the top N recommendations
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

    # Extract movie IDs from recommendations
    recommended_movie_ids = [int(prediction.iid) for prediction in top_n]

    # Return recommended movie IDs
    return jsonify({'recommendations': recommended_movie_ids})


if __name__ == '__main__':
    app.run(debug=True, port=4080)

