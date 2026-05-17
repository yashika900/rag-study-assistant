"""Local sentence-transformers embedding service."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_sentence_transformer() -> SentenceTransformer:
    """Load the local embedding model once and reuse it."""

    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper around sentence-transformers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for document chunks."""

        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        """Create an embedding for one user question."""

        return embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model() -> LocalSentenceTransformerEmbeddings:
    """Return the reusable local embedding wrapper used by Chroma."""

    return LocalSentenceTransformerEmbeddings()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate local embeddings for many text chunks."""

    if not texts:
        raise ValueError("No texts were provided for embedding.")

    print("Local embeddings started")
    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL_NAME}")
    model = get_sentence_transformer()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"Local embeddings complete. Generated {len(embeddings)} embeddings")
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Generate a local embedding for one query."""

    if not text.strip():
        raise ValueError("Query text cannot be empty.")

    print("Local query embedding started")
    model = get_sentence_transformer()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print("Local query embedding complete")
    return embedding.tolist()
