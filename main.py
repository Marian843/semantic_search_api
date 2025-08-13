from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# ====== CONFIG ======
SAVE_DIR = "data"  # folder where you saved from Colab
INDEX_FILE = os.path.join(SAVE_DIR, "faiss_index.bin")
META_FILE = os.path.join(SAVE_DIR, "metadata.npy")
MODEL_NAME = "all-MiniLM-L6-v2"

# ====== LOAD MODEL ======
print("Loading SBERT model...")
model = SentenceTransformer(MODEL_NAME)

# ====== LOAD FAISS & METADATA ======
print("Loading FAISS index and metadata...")
if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
    raise FileNotFoundError("Index or metadata file not found in data/ folder.")

index = faiss.read_index(INDEX_FILE)

metadata = np.load(META_FILE, allow_pickle=True)
print(f"Metadata loaded: {len(metadata)} entries")
print("First entry type:", type(metadata[0]))

if index.ntotal != len(metadata):
    raise ValueError(f"Index size ({index.ntotal}) and metadata size ({len(metadata)}) do not match.")

# ====== FASTAPI APP ======
app = FastAPI(title="Semantic Search API", version="1.0")


class SearchResult(BaseModel):
    title: str
    abstract: str
    topic: str
    score: float


@app.get("/search", response_model=List[SearchResult])
def search(query: str = Query(..., description="Search query text"), top_k: int = 5):
    """Search papers using FAISS index and SBERT embeddings"""
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        entry = metadata[idx]
        # unwrap if needed
        if not isinstance(entry, dict):
            entry = entry.item()

        results.append(SearchResult(
            title=entry.get("titles", ""),
            abstract=entry.get("summaries", ""),
            topic=entry.get("primary_topic", ""),
            score=float(dist)
        ))
    return results


@app.get("/")
def root():
    return {"message": "Semantic Search API is running"}