# MovieLens Recommender System - Backend

Flask API for the MovieLens movie recommendation system.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place required files:
- `models/cuml_random_forest_regressor.joblib`
- `models/knn_movie_model.pkl`
- `data/df.csv`

4. Run the server:
```bash
python app.py
```

Server will start at `http://localhost:5000`

## API Endpoints

### GET /api/health
Health check

### POST /api/recommendations
Get personalized recommendations
```json
{
  "genres": ["Action", "Thriller"],
  "age_preference": "recent",
  "popularity": "popular",
  "min_rating": 4.0,
  "num_recommendations": 10
}
```

### GET /api/search?query=movie
Search movies by title

### GET /api/similar/{movie_id}
Get similar movies using KNN

### GET /api/insights
Get analytics data for dashboard

### GET /api/movie/{movie_id}
Get specific movie details

To Run:

Create the folder structure as shown
Copy all the code files into their respective locations
Place your models and data files in the correct folders
Install dependencies: pip install -r requirements.txt
Run: python app.py