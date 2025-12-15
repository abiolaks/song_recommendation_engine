# Song Recommendation Engine (Cosine Similarity)

## Overview

This project implements a **content-based song recommendation system** using **cosine similarity**.
Given a song a user likes, the system recommends similar songs based on **artist name, song title, and lyrics text**.

The system is designed to be:

* Memory-safe (no full similarity matrix)
* Real-time capable
* Scalable for datasets with tens of thousands of songs
* Deployable as a REST API using FastAPI

The recommendation logic compares a single song vector against all other song vectors using cosine similarity, avoiding the O(N²) memory cost that caused earlier failures.

---

## Dataset

* **Number of songs:** ~57,000
* **Features used:**

  * `artist`
  * `song`
  * `text` (lyrics)

The average lyrics length is approximately 219 words, which is suitable for TF-IDF–based text modeling with dimensionality reduction.

---

## Recommendation Approach

### Feature Engineering

The following text features are combined into a single field:

* Artist name
* Song title
* Lyrics text

This combined text is used as input for vectorization.

---

### Vectorization and Dimensionality Reduction (Offline)

1. **TF-IDF Vectorization**

   * Vocabulary size is capped to control memory usage.
   * Unigrams and bigrams are used.

2. **Truncated SVD**

   * Reduces high-dimensional TF-IDF vectors to dense semantic vectors.
   * Default dimensionality: 200 components.

3. **L2 Normalization**

   * All vectors are normalized so cosine similarity can be computed efficiently using dot products.

These steps are executed **offline** and saved to disk.

---

### Similarity Computation (Online)

* Cosine similarity is computed **only between one song and all other songs**.
* No full song-to-song similarity matrix is created.
* Similarity scores are computed using a dot product between normalized vectors.

This ensures:

* O(N) memory usage
* Fast real-time recommendations
* No large memory allocation errors

---

## API Architecture

The system is exposed via a FastAPI service.

### Key Characteristics

* Models and vectors are loaded once at startup
* No retraining or re-vectorization during inference
* Real-time response latency (milliseconds)

---

## API Endpoints

### Root Endpoint

**GET /**

Returns a simple health message.

**Response**

```json
{
  "message": "Song Recommender API is running"
}
```

---

### Recommendation Endpoint

**POST /recommendations**

Returns songs similar to the input song using cosine similarity.

**Request Body**

```json
{
  "song": "Siberia",
  "k": 10
}
```

**Parameters**

* `song` (string): Song title provided by the user
* `k` (integer, optional): Number of recommendations (default: 10)

**Response**

```json
{
  "input_song": "Siberia",
  "matched_song": "Siberia",
  "recommendations": [
    {
      "song": "Cold Desert",
      "artist": "Kings of Leon"
    },
    {
      "song": "Holocene",
      "artist": "Bon Iver"
    }
  ]
}
```

---

## Fuzzy Matching

To handle typos and partial song titles:

* Fuzzy string matching is applied to user input
* The closest matching song title in the dataset is used as the query

If no suitable match is found, the API returns a 404 error.

---

## Project Structure

```
song_recommendation_engine/
│
├── main.py
├── sportify_millsongs.csv
├── song_vectors_reduced.pkl
├── requirements.txt
├── README.md
```

---

## Running the Project Locally

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

---

### 2. Start the API Server

```bash
uvicorn main:app --reload
```

The server will run at:

```
http://127.0.0.1:8000
```

---

### 3. Test the API

Open the Swagger UI:

```
http://127.0.0.1:8000/docs
```

Use the `/recommendations` endpoint to test song recommendations.

---

## Performance Characteristics

* **Startup time:** A few seconds
* **Recommendation latency:** ~10–40 ms
* **Memory usage:** Less than 1 GB
* **Scalability:** Suitable for tens of thousands of songs

---

## Why Cosine Similarity Is Used

* Cosine similarity is well-suited for high-dimensional text embeddings
* It measures semantic similarity independent of text length
* When applied correctly (single-vector comparison), it is efficient and safe

The project avoids computing a full similarity matrix, which would require tens of gigabytes of memory.

---

## Limitations and Future Improvements

* Song title matching is string-based; using song IDs would be more robust
* Lyrics-only similarity may miss musical features (tempo, mood, genre)
* Can be extended with:

  * Artist-weighted scoring
  * Popularity-based ranking
  * Sentence-transformer embeddings
  * Approximate nearest neighbor search (FAISS)

---

## License

This project is for educational and demonstration purposes.

