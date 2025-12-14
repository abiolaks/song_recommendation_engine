# import the libraries
import numpy as np
import pandas as pd
import difflib  # for comparing sequences
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # for converting text to matrix of TF-IDF features
from sklearn.metrics.pairwise import (
    cosine_similarity,
)  # to compute cosine similarity between samples in two matrices
from sklearn.decomposition import TruncatedSVD  # for dimensionality reduction
from sklearn.neighbors import NearestNeighbors  # for nearest neighbor search
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import difflib  # for finding close matches - fuzzy matching
from sklearn.preprocessing import normalize # for normalizing vectors
import joblib


app = FastAPI()
app.add_middleware(
    CORSMiddleware,  # to access api from anywhere
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# read data
song_data = pd.read_csv("./spotify_millsongdata.csv")


# Selecting features for the recommendation engine
features = ["artist", "song", "text"]  # this are the 3 features of interest


# Data preprocessing steps below

# replacing null values with null string - addressing missing values
for feature in features:
    song_data[feature] = song_data[feature].fillna("")


# Creating a new combined features column - to one text column for tfidf vectorizer
combined_features = song_data[features].apply(lambda row: " ".join(row), axis=1)

# vectorizing the combined features using bounded TFIDF -
# this is important has the dataset is large over 57k.
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=25_000,  # reduced due to moderate text length
    min_df=3,
    max_df=0.85,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

feature_vectors = tfidf.fit_transform(combined_features)


# this is high dimension matrix - lets reduce the dimension using dimensionality reduction techniques


svd = TruncatedSVD(n_components=200, random_state=42)


# The above compresses tens of thousands of sparse features into dense semantic vectors.

# This reduces the feature vectors to a more manageable size while retaining most of the important information.

feature_vectors_reduced = svd.fit_transform(feature_vectors)
# save as joblib object
joblib.dump(feature_vectors_reduced, "song_vectors_reduced.pkl")


# normalizing the reduced feature vectors
# this step ensures that the vectors have unit length,
# This enables fast cosine similarity via dot product.
X_norm = normalize(feature_vectors_reduced, norm="l2")


# Compute the cosine similarity for one song against all songs - to avoid crashing memory
# function to get song recommendations
def recommend_with_cosine(song_index, k=10):
    query_vec = X_norm[song_index]           # shape: (200,)
    scores = np.dot(X_norm, query_vec)       # shape: (57k,)

    top_indices = np.argsort(scores)[-k-1:-1][::-1]
    return song_data.iloc[top_indices][["song", "artist"]]



def find_best_match(song_title: str):
    titles = song_data["song"].str.lower().tolist()
    matches = difflib.get_close_matches(song_title.lower(), titles, n=1, cutoff=0.6)
    return matches[0] if matches else None


# getting user input
class SongRequest(BaseModel):
    song_title: str
    k: int = 10


app = FastAPI(title="AI Song Recommender API")


@app.get("/")
def root():
    return {"message": "Welcome to the AI Song Recommender API"}


# Load song data
song_data = pd.read_csv("songs.csv") # make sure to have this csv file in the same directory. this is the database of songs
@app.post("/recommendations")
def recommend_song(request: SongRequest):

    best_match = find_best_match(request.song)
    if not best_match:
        raise HTTPException(status_code=404, detail="Song not found")

    song_index = song_data[song_data.song.str.lower() == best_match].index[0]

    results = recommend_with_cosine(song_index, request.k)

    return {
        "input_song": request.song,
        "matched_song": best_match,
        "recommendations": results.to_dict(orient="records")
    }

