"""Gemini embedding service — drop-in replacement for sentence-transformers.

Switches from local sentence-transformers (requires ~1.2 GB RAM with torch)
to Google's Gemini embedding-001 API (zero local RAM, same vector quality).
The public interface is IDENTICAL to the old file so nothing else changes.
"""

import os
import time
from functools import lru_cache

import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "models/embedding-001"
_BATCH_SIZE = 50          # Gemini allows up to 100 per batch call
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0        # seconds between retries


def _configure_genai() -> None:
    """Configure the Gemini SDK once using the env variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or Render environment variables."
        )
    genai.configure(api_key=api_key)


# ---------------------------------------------------------------------------
# Core embedding helpers
# ---------------------------------------------------------------------------

def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Call Gemini embedding API for one batch with retry logic."""
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL_NAME,
                content=texts,
                task_type=task_type,
            )
            # Gemini returns a dict with key "embedding" for single input
            # and a list under "embedding" for batch input.
            raw = result["embedding"]
            # Single string input → list[float]; batch → list[list[float]]
            if raw and isinstance(raw[0], float):
                return [raw]
            return raw
        except Exception as exc:  # noqa: BLE001
            if attempt == _RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"Gemini embedding failed after {_RETRY_ATTEMPTS} attempts: {exc}"
                ) from exc
            print(f"Embedding attempt {attempt} failed ({exc}), retrying…")
            time.sleep(_RETRY_DELAY * attempt)

    return []  # unreachable, satisfies mypy


# ---------------------------------------------------------------------------
# Public API — same names as the old sentence-transformers version
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate Gemini embeddings for many text chunks (document side)."""
    if not texts:
        raise ValueError("No texts were provided for embedding.")

    _configure_genai()
    print("Gemini embeddings started")
    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL_NAME}")

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        batch_embeddings = _embed_batch(batch, task_type="retrieval_document")
        all_embeddings.extend(batch_embeddings)
        print(f"  Processed {min(i + _BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    print(f"Gemini embeddings complete. Generated {len(all_embeddings)} embeddings")
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Generate a Gemini embedding for one user query."""
    if not text.strip():
        raise ValueError("Query text cannot be empty.")

    _configure_genai()
    print("Gemini query embedding started")

    result = _embed_batch([text], task_type="retrieval_query")

    print("Gemini query embedding complete")
    return result[0]


# ---------------------------------------------------------------------------
# LangChain-compatible wrapper (same class name as before)
# ---------------------------------------------------------------------------

class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper — now backed by Gemini instead of torch.

    The class name is intentionally kept the same so that vector_store.py
    and any other file that imports it requires zero changes.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for document chunks."""
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        """Create an embedding for one user question."""
        return embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model() -> LocalSentenceTransformerEmbeddings:
    """Return the reusable embedding wrapper used by Chroma.

    lru_cache ensures the object is created only once per process,
    mirroring the singleton behaviour of the old sentence-transformer loader.
    """
    print(f"Initialising Gemini embedding model: {EMBEDDING_MODEL_NAME}")
    return LocalSentenceTransformerEmbeddings()


# ---------------------------------------------------------------------------
# Kept for backward compatibility — some files may import this directly
# ---------------------------------------------------------------------------

def get_sentence_transformer():  # noqa: ANN201
    """Backward-compat shim — returns the Gemini wrapper instead of a torch model."""
    return get_embedding_model()