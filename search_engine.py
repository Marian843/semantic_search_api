import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_DIR, "papers.index")
META_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load metadata
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

titles = metadata["titles"]
summaries = metadata["summaries"]
topics = metadata["topics"]

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "title": titles[idx],
            "summary": summaries[idx],
            "topic": topics[idx],
            "score": float(distances[0][i])
        })
    return results
