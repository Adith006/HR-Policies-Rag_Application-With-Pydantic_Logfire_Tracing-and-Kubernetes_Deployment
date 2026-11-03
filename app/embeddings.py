from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load one shared CPU embedding model for the RAG application."""
    return SentenceTransformer(MODEL_NAME, device="cpu")
