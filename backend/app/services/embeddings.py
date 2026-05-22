"""Gemini embedding service using the new google-genai package.

Replaces sentence-transformers (too heavy for Render free tier)
with Google's Gemini text-embedding-004 API.
Public interface is identical so no other files need changes.
"""

import os
import time
from functools import lru_cache

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "text-embedding-004"
_BATCH_SIZE = 50
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0


def _get_client() -> genai.Client:
    """Create a Gemini client using the API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or Render environment variables."
        )
    return genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1"},
    )


# ---------------------------------------------------------------------------
# Core embedding helpers
# ---------------------------------------------------------------------------

def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Call Gemini embedding API for one batch with retry logic."""
    client = _get_client()

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in response.embeddings]
        except Exception as exc:
            if attempt == _RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"Gemini embedding failed after {_RETRY_ATTEMPTS} attempts: {exc}"
                ) from exc
            print(f"Embedding attempt {attempt} failed ({exc}), retrying...")
            time.sleep(_RETRY_DELAY * attempt)

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate Gemini embeddings for many text chunks (document side)."""
    if not texts:
        raise ValueError("No texts were provided for embedding.")

    print("Gemini embeddings started")
    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL_NAME}")

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i: i + _BATCH_SIZE]
        batch_embeddings = _embed_batch(batch, task_type="RETRIEVAL_DOCUMENT")
        all_embeddings.extend(batch_embeddings)
        print(f"  Processed {min(i + _BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    print(f"Gemini embeddings complete. Generated {len(all_embeddings)} embeddings")
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Generate a Gemini embedding for one user query."""
    if not text.strip():
        raise ValueError("Query text cannot be empty.")

    print("Gemini query embedding started")
    result = _embed_batch([text], task_type="RETRIEVAL_QUERY")
    print("Gemini query embedding complete")
    return result[0]


# ---------------------------------------------------------------------------
# LangChain-compatible wrapper (class name unchanged)
# ---------------------------------------------------------------------------

class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper — backed by Gemini text-embedding-004.

    Class name kept identical so vector_store.py needs zero changes.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model() -> LocalSentenceTransformerEmbeddings:
    """Return the reusable embedding wrapper used by Chroma."""
    print(f"Initialising Gemini embedding model: {EMBEDDING_MODEL_NAME}")
    return LocalSentenceTransformerEmbeddings()



def get_sentence_transformer() -> LocalSentenceTransformerEmbeddings:
    """Backward-compat shim."""
    return get_embedding_model()