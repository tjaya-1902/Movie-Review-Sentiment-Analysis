from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import joblib
import requests
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"])

# Load sentiment model and vectorizer
model = joblib.load("backend/sentiment_model.pkl")
vectorizer = joblib.load("backend/tfidf_vectorizer.pkl")

# Global state to track last movie and review index
review_tracker = {
    "last_title": "",
    "current_index": 0,
    "cached_reviews": []
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    vector = vectorizer.transform([review])
    prediction = model.predict(vector)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"sentiment": sentiment})


@app.route("/get_movie_info", methods=["POST"])
def get_movie_info():
    data = request.get_json()
    movie_title = data.get("title", "").strip()

    API_KEY = os.getenv("TMDB_API_KEY")
    if not API_KEY:
        return jsonify({"error": "TMDb API key not set"}), 500

    try:
        # Get movie ID
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        search_response = requests.get(search_url).json()
        if not search_response["results"]:
            return jsonify({"error": "Movie not found"}), 404

        movie = search_response["results"][0]
        movie_id = movie["id"]

        # Get detailed movie info
        detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        detail_response = requests.get(detail_url).json()

        # Get credits for director and cast
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
        credits = requests.get(credits_url).json()

        # Get director
        crew = credits.get("crew", [])
        director = next((person["name"] for person in crew if person["job"] == "Director"), "N/A")

        # Get actors
        cast = credits.get("cast", [])
        actors = [actor["name"] for actor in cast[:3]]

        movie_info = {
            "title": detail_response.get("title"),
            "overview": detail_response.get("overview"),
            "release_date": detail_response.get("release_date"),
            "rating": detail_response.get("vote_average"),
            "genres": [g["name"] for g in detail_response.get("genres", [])],
            "poster_url": f"https://image.tmdb.org/t/p/w500{detail_response.get('poster_path')}" if detail_response.get("poster_path") else None,
            "director": director,
            "actors": actors
        }



        return jsonify(movie_info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_reviews", methods=["POST"])
def get_reviews():
    data = request.get_json()
    movie_title = data.get("title", "").strip()

    API_KEY = os.getenv("TMDB_API_KEY")
    if not API_KEY:
        return jsonify({"error": "TMDb API key not set"}), 500

    global review_tracker

    try:
        # If title changed or no cache, reset state
        if movie_title != review_tracker["last_title"]:
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
            search_response = requests.get(search_url).json()

            if not search_response["results"]:
                return jsonify({"error": "Movie not found"}), 404

            movie_id = search_response["results"][0]["id"]
            review_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}"
            review_response = requests.get(review_url).json()

            reviews = review_response.get("results", [])
            review_tracker = {
                "last_title": movie_title,
                "current_index": 0,
                "cached_reviews": [r["content"] for r in reviews]
            }

        # Get next review
        reviews = review_tracker["cached_reviews"]
        index = review_tracker["current_index"]

        if not reviews:
            return jsonify({"review": "No reviews available for this movie."})

        review = reviews[index % len(reviews)]
        review_tracker["current_index"] += 1

        return jsonify({"review": review})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)