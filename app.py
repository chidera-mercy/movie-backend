from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import requests
from functools import lru_cache

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:*",
            "https://movie-frontend-six-rust.vercel.app",
            "https://*.vercel.app"  # Allow all Vercel domains
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ==================== TMDB API CONFIGURATION ====================
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Your API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# ==================== CONFIGURATION ====================
# Model files
SIMILARITY_MATRIX_PATH = "./models/final_recommender_model.pkl"
RF_MODEL_PATH = "./models/rf_content_based_model.pkl"
SCALER_PATH = "./models/scaler.pkl"
BASELINE_PATH = "./models/baseline_topN.csv"

# Data files
MOVIES_DATA_PATH = "./data/movies_for_deployment.csv"
LINKS_PATH = "./data/links.csv"
ORIGINAL_MOVIES_PATH = "./data/movies.csv"

# Insights directory
INSIGHTS_DIR = "./insights"

# Content scaler path
CONTENT_SCALER_PATH = "./data/content_scaler.pkl"

# ==================== LOAD MODELS & DATA ====================
print("Loading models and data...")

# Load Cosine Similarity Matrix for similar movies
similarity_matrix = joblib.load(SIMILARITY_MATRIX_PATH)
print(f"✅ Loaded similarity matrix with shape: {similarity_matrix.shape}")

# Load Random Forest model
rf_model = joblib.load(RF_MODEL_PATH)

# Load content scaler (for 6 numerical movie features only)
content_scaler = joblib.load(CONTENT_SCALER_PATH)
print("✅ Loaded content scaler")

# Content numerical features that need scaling (6 features)
CONTENT_NUMERICAL = ['year', 'movie_age', 'num_genres', 'avg_rating', 'num_ratings', 'rating_std']

# Load movie data
movies_df = pd.read_csv(MOVIES_DATA_PATH)

# Load links for TMDB poster support
links_df = pd.read_csv(LINKS_PATH)
movies_df = movies_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')

# Load baseline top movies
baseline_top = pd.read_csv(BASELINE_PATH)

print("✅ All models and data loaded successfully!")

# ==================== HELPER FUNCTIONS ====================

@lru_cache(maxsize=1000)
def get_tmdb_poster_url(tmdb_id):
    """
    Fetch poster URL from TMDB API using the movie ID.
    Uses caching to avoid repeated API calls for the same movie.
    """
    if pd.isna(tmdb_id):
        return None
    
    try:
        tmdb_id = int(tmdb_id)
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {"api_key": TMDB_API_KEY}
        
        response = requests.get(url, params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        
        return None
        
    except Exception as e:
        print(f"Error fetching poster for tmdbId {tmdb_id}: {e}")
        return None

def encode_movie_age_bin(age_bin_value):
    """
    Encode movie_age_bin as a single numerical value.
    This matches how it was encoded during training.
    """
    bin_mapping = {
        'Classic': 0,
        'Vintage': 1, 
        'Modern': 2,
        'Recent': 3
    }
    return bin_mapping.get(age_bin_value, 2)  # Default to Modern if unknown

def prepare_movie_features(movie_row):
    """
    Prepare 39 features for RF prediction (matches training exactly).
    
    Expected features (in order):
    - First 6 features (SCALED): 'year', 'movie_age', 'num_genres', 'avg_rating', 'num_ratings', 'rating_std'
    - Next 33 features (NOT SCALED):
        'rating_year', 'rating_month', 'rating_day_of_week', 'rating_hour',
        'movie_popularity_percentile', 'is_classic', 'is_recent', 
        'is_popular_movie', 'is_obscure_movie',
        19 genre columns,
        'rating_month_sin', 'rating_month_cos', 'rating_day_sin', 'rating_day_cos',
        'movie_age_bin_encoded'
    
    Note: Rating time features use current date as default since we don't have actual rating events.
    """
    from datetime import datetime
    import math
    
    # Current date for default rating time features
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    current_day_of_week = now.weekday()  # Monday=0, Sunday=6
    current_hour = 18  # Default to 6 PM (typical movie watching time)
    
    # Cyclical encoding for month and day
    month_sin = math.sin(2 * math.pi * current_month / 12)
    month_cos = math.cos(2 * math.pi * current_month / 12)
    day_sin = math.sin(2 * math.pi * current_day_of_week / 7)
    day_cos = math.cos(2 * math.pi * current_day_of_week / 7)
    
    # First 6 features - NEED SCALING (content features)
    content_features = np.array([
        movie_row['year'],
        movie_row['movie_age'],
        movie_row['num_genres'],
        movie_row['avg_rating'],
        movie_row['num_ratings'],
        movie_row['rating_std']
    ]).reshape(1, -1)
    
    # Remaining 33 features - NO SCALING NEEDED
    other_features = [
        current_year,  # rating_year
        current_month,  # rating_month
        current_day_of_week,  # rating_day_of_week
        current_hour,  # rating_hour
        movie_row['movie_popularity_percentile'],
        movie_row['is_classic'],
        movie_row['is_recent'],
        movie_row['is_popular_movie'],
        movie_row['is_obscure_movie'],
        # 19 genre features (exact order from model)
        movie_row['genre_Drama'],
        movie_row['genre_Comedy'],
        movie_row['genre_Action'],
        movie_row['genre_Thriller'],
        movie_row['genre_Adventure'],
        movie_row['genre_Sci-Fi'],
        movie_row['genre_Romance'],
        movie_row['genre_Crime'],
        movie_row['genre_Fantasy'],
        movie_row['genre_Children'],
        movie_row['genre_Mystery'],
        movie_row['genre_Horror'],
        movie_row['genre_Animation'],
        movie_row['genre_War'],
        movie_row['genre_IMAX'],
        movie_row['genre_Musical'],
        movie_row['genre_Western'],
        movie_row['genre_Documentary'],
        movie_row['genre_Film-Noir'],
        # Cyclical time features
        month_sin,
        month_cos,
        day_sin,
        day_cos,
        # Movie age bin encoded
        encode_movie_age_bin(movie_row['movie_age_bin'])
    ]
    
    return content_features, np.array(other_features).reshape(1, -1)

def recommend_similar_movies(movie_id, top_k=10):
    """
    Find K most similar movies using cosine similarity matrix.
    
    Parameters:
    -----------
    movie_id : int
        The ID of the movie to find similar movies for
    top_k : int
        Number of similar movies to return
        
    Returns:
    --------
    list : List of similar movie IDs (ordered by similarity), or empty list if movie not found
    """
    try:
        # Check if movie exists in similarity matrix
        if movie_id not in similarity_matrix.index:
            print(f"Movie ID {movie_id} not found in similarity matrix")
            return []
        
        # Get similarity scores for this movie
        sim_scores = similarity_matrix[movie_id].sort_values(ascending=False)
        
        # Get top K similar movies (excluding the movie itself at index 0)
        similar_movie_ids = sim_scores.iloc[1:top_k+1].index.tolist()
        
        return similar_movie_ids
        
    except Exception as e:
        print(f"Error in recommend_similar_movies for movie_id {movie_id}: {e}")
        return []

# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({"status": "healthy", "message": "API is running!"})

@app.route('/api/search-movies', methods=['GET'])
def search_movies():
    """Search movies by title (autocomplete)"""
    query = request.args.get('query', '').lower()
    
    if len(query) < 2:
        return jsonify([])
    
    # Search in titles
    matches = movies_df[
        movies_df['title'].str.lower().str.contains(query, na=False)
    ].head(10)
    
    results = []
    for _, movie in matches.iterrows():
        results.append({
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'year': int(movie['year']) if pd.notna(movie['year']) else None,
            'avgRating': round(float(movie['avg_rating']), 2)
        })
    
    return jsonify(results)

@app.route('/api/similar-movies', methods=['POST'])
def get_similar_movies():
    """Get similar movies based on selected movie"""
    data = request.json
    movie_id = data.get('movieId')
    top_k = data.get('topK', 10)
    
    if not movie_id:
        return jsonify({"error": "movieId is required"}), 400
    
    # Check if movie exists in our database
    movie_exists = movies_df[movies_df['movieId'] == movie_id]
    if movie_exists.empty:
        return jsonify({"error": "Movie not found in database"}), 404
    
    # Get similar movie IDs
    similar_ids = recommend_similar_movies(movie_id, top_k=top_k)
    
    if not similar_ids:
        movie_title = movie_exists.iloc[0]['title']
        return jsonify({
            "error": f"No genome data available for '{movie_title}'. This movie doesn't have enough tag/genre information to find similar movies.",
            "recommendations": []
        }), 404
    
    # Get movie details
    similar_movies = movies_df[movies_df['movieId'].isin(similar_ids)]
    
    results = []
    for movie_id in similar_ids:  # Maintain order
        movie_data = similar_movies[similar_movies['movieId'] == movie_id]
        if movie_data.empty:
            continue
            
        movie = movie_data.iloc[0]
        results.append({
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'year': int(movie['year']) if pd.notna(movie['year']) else None,
            'avgRating': round(float(movie['avg_rating']), 2),
            'numRatings': int(movie['num_ratings']),
            'posterUrl': get_tmdb_poster_url(movie.get('tmdbId'))
        })
    
    return jsonify(results)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Get personalized recommendations based on user preferences.
    Expected JSON body:
    {
        "genres": ["Action", "Sci-Fi"],
        "agePreference": "recent" | "modern" | "vintage" | "classic" | "",
        "popularity": "popular" | "hidden_gems" | "",
        "topN": 10,
        "minRating": 4.0 (optional)
    }
    """
    data = request.json
    
    # Extract preferences
    selected_genres = data.get('genres', [])
    age_pref = data.get('agePreference', '')
    popularity_pref = data.get('popularity', '')
    top_n = data.get('topN', 10)
    min_rating = data.get('minRating', None)
    
    # Start with all movies
    filtered_movies = movies_df.copy()
    
    # Filter by genres (if any selected)
    if selected_genres:
        genre_filter = pd.Series([False] * len(filtered_movies))
        for genre in selected_genres:
            genre_col = f"genre_{genre}"
            if genre_col in filtered_movies.columns:
                genre_filter = genre_filter | (filtered_movies[genre_col] == 1)
        filtered_movies = filtered_movies[genre_filter]
    
    # Filter by age preference
    if age_pref == 'recent':
        filtered_movies = filtered_movies[filtered_movies['is_recent'] == 1]
    elif age_pref == 'classic':
        filtered_movies = filtered_movies[filtered_movies['is_classic'] == 1]
    elif age_pref == 'modern':
        filtered_movies = filtered_movies[
            (filtered_movies['year'] >= 1995) & (filtered_movies['year'] <= 2015)
        ]
    elif age_pref == 'vintage':
        filtered_movies = filtered_movies[
            (filtered_movies['year'] >= 1970) & (filtered_movies['year'] < 1995)
        ]
    
    # Filter by popularity
    if popularity_pref == 'popular':
        filtered_movies = filtered_movies[filtered_movies['is_popular_movie'] == 1]
    elif popularity_pref == 'hidden_gems':
        filtered_movies = filtered_movies[filtered_movies['is_obscure_movie'] == 1]
    
    # Filter by minimum rating
    if min_rating:
        filtered_movies = filtered_movies[filtered_movies['avg_rating'] >= min_rating]
    
    # Check if we have enough movies
    if len(filtered_movies) < 1:
        return jsonify({
            "error": "No movies match your preferences. Try relaxing some filters.",
            "recommendations": []
        }), 404
    
    # Prepare features for prediction
    predictions = []
    for _, movie in filtered_movies.iterrows():
        try:
            # Get features split: 6 content features + 33 other features
            content_features, other_features = prepare_movie_features(movie)
            
            # Scale only the 6 content features
            content_scaled = content_scaler.transform(content_features)
            
            # Concatenate: scaled content (6) + unscaled others (33) = 39 total
            features_final = np.concatenate([content_scaled, other_features], axis=1)
            
            # Predict rating
            pred_rating = rf_model.predict(features_final)[0]
            
            predictions.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'year': int(movie['year']) if pd.notna(movie['year']) else None,
                'avgRating': round(float(movie['avg_rating']), 2),
                'numRatings': int(movie['num_ratings']),
                'predictedRating': round(float(pred_rating), 2),
                'posterUrl': get_tmdb_poster_url(movie.get('tmdbId'))
            })
        except Exception as e:
            print(f"Error predicting for movie {movie['movieId']}: {e}")
            continue
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x['predictedRating'], reverse=True)
    
    # Return top N
    return jsonify(predictions[:top_n])

@app.route('/api/baseline-top', methods=['GET'])
def get_baseline_top():
    """Get baseline top-rated movies (weighted popularity)"""
    top_n = int(request.args.get('topN', 20))
    
    results = []
    for _, movie in baseline_top.head(top_n).iterrows():
        # Get poster from main df
        movie_data = movies_df[movies_df['movieId'] == movie['movieId']]
        poster_url = None
        if not movie_data.empty:
            poster_url = get_tmdb_poster_url(movie_data.iloc[0].get('tmdbId'))
        
        results.append({
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'weightedScore': round(float(movie['weighted_score']), 2),
            'avgRating': round(float(movie['avg_rating']), 2),
            'numRatings': int(movie['num_ratings']),
            'posterUrl': poster_url
        })
    
    return jsonify(results)

@app.route('/api/insights/stats', methods=['GET'])
def get_insights_stats():
    """Get basic statistics for insights dashboard"""
    stats = {
        'totalMovies': int(len(movies_df)),
        'avgRating': round(float(movies_df['avg_rating'].mean()), 2),
        'totalRatings': int(movies_df['num_ratings'].sum()),
        'mostPopularGenre': get_most_popular_genre(),
        'newestMovie': int(movies_df['year'].max()),
        'oldestMovie': int(movies_df['year'].min())
    }
    
    return jsonify(stats)

def get_most_popular_genre():
    """Find the most popular genre by number of movies"""
    genre_cols = [col for col in movies_df.columns if col.startswith('genre_')]
    genre_counts = {}
    
    for col in genre_cols:
        genre_name = col.replace('genre_', '')
        genre_counts[genre_name] = int(movies_df[col].sum())
    
    most_popular = max(genre_counts, key=genre_counts.get)
    return most_popular

@app.route('/api/insights/charts', methods=['GET'])
def get_insights_charts():
    """Get data for charts in insights dashboard"""
    
    # Rating distribution
    rating_dist = movies_df.groupby(
        pd.cut(movies_df['avg_rating'], bins=[0, 1, 2, 3, 4, 5])
    ).size().to_dict()
    
    rating_categories = {
        '0-1': rating_dist.get(pd.Interval(0, 1, closed='right'), 0),
        '1-2': rating_dist.get(pd.Interval(1, 2, closed='right'), 0),
        '2-3': rating_dist.get(pd.Interval(2, 3, closed='right'), 0),
        '3-4': rating_dist.get(pd.Interval(3, 4, closed='right'), 0),
        '4-5': rating_dist.get(pd.Interval(4, 5, closed='right'), 0)
    }
    
    # Genre distribution (top 10)
    genre_cols = [col for col in movies_df.columns if col.startswith('genre_')]
    genre_counts = {}
    
    for col in genre_cols:
        genre_name = col.replace('genre_', '')
        genre_counts[genre_name] = int(movies_df[col].sum())
    
    # Sort and get top 10
    top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Movies by rating category
    high_rated = len(movies_df[movies_df['avg_rating'] >= 4.0])
    medium_rated = len(movies_df[(movies_df['avg_rating'] >= 3.0) & (movies_df['avg_rating'] < 4.0)])
    low_rated = len(movies_df[movies_df['avg_rating'] < 3.0])
    
    return jsonify({
        'ratingDistribution': rating_categories,
        'topGenres': top_genres,
        'ratingCategories': {
            'High (4.0+)': high_rated,
            'Medium (3.0-4.0)': medium_rated,
            'Low (<3.0)': low_rated
        }
    })

@app.route('/api/insights/images', methods=['GET'])
def get_insights_images():
    """Get list of available insight images"""
    try:
        # List all PNG files in insights directory
        if os.path.exists(INSIGHTS_DIR):
            images = [f for f in os.listdir(INSIGHTS_DIR) if f.endswith('.png')]
            images.sort()  # Sort alphabetically
            return jsonify({'images': images})
        else:
            return jsonify({'images': [], 'error': 'Insights directory not found'})
    except Exception as e:
        return jsonify({'images': [], 'error': str(e)})

@app.route('/insights/<filename>')
def serve_insight_image(filename):
    """Serve insight images"""
    try:
        return send_from_directory(INSIGHTS_DIR, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

# ==================== RUN SERVER ====================
if __name__ == '__main__':    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
