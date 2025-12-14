# import the libraries
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import difflib  # for finding close matches - fuzzy matching
from sklearn.preprocessing import normalize  # for normalizing vectors
import joblib


app = FastAPI()
app.add_middleware(
    CORSMiddleware,  # to access api from anywhere
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Load data & vectors
# -----------------------
song_data = pd.read_csv("./spotify_millsongdata.csv")

X_reduced = joblib.load("song_vectors_reduced.pkl")
X_norm = normalize(X_reduced, norm="l2")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Song Recommender API")


class SongRequest(BaseModel):
    song: str
    k: int = 10


@app.get("/")
def root():
    return {"message": "Song Recommender API is running"}


# -----------------------
# Helper functions
# -----------------------
def find_best_match(song_title: str):
    titles = song_data["song"].str.lower().tolist()
    matches = difflib.get_close_matches(song_title.lower(), titles, n=1, cutoff=0.6)
    return matches[0] if matches else None


def recommend_with_cosine(song_index: int, k: int = 10):
    query_vec = X_norm[song_index]
    scores = np.dot(X_norm, query_vec)
    top_indices = np.argsort(scores)[-k - 1 : -1][::-1]
    return song_data.iloc[top_indices][["song", "artist"]]


# -----------------------
# API endpoint
# -----------------------
@app.post("/recommendations")
def recommend_song(request: SongRequest):
    best_match = find_best_match(request.song)

    if not best_match:
        raise HTTPException(status_code=404, detail="Song not found")

    song_index = song_data[song_data["song"].str.lower() == best_match].index[0]

    recs = recommend_with_cosine(song_index, request.k)

    return {
        "input_song": request.song,
        "matched_song": best_match,
        "recommendations": recs.to_dict(orient="records"),
    }
