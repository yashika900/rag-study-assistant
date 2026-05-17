"""Vector storage service.

Chroma is the intended backend. On Windows + Python 3.12, current Chroma wheels
can crash in native code during inserts, so this module includes a tiny SQLite
fallback that keeps the MVP usable while preserving the same service interface.
"""

import json
import math
import os
import platform
import shutil
import sqlite3
import sys
import traceback
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.app.services.embeddings import embed_query, embed_texts, get_embedding_model
from backend.app.utils.helpers import CHROMA_DIR


COLLECTION_NAME = "study_material"
SIMPLE_STORE_PATH = CHROMA_DIR / "simple_vector_store.sqlite"


def _use_simple_store() -> bool:
    """Choose the local SQLite fallback when Chroma is unsafe locally."""

    backend = os.getenv("VECTOR_STORE_BACKEND", "").strip().lower()
    if backend == "chroma":
        return False
    if backend == "simple":
        return True

    return platform.system() == "Windows" and sys.version_info >= (3, 12)


def reset_vector_store() -> None:
    """Remove existing embeddings so each upload becomes the active study set."""

    try:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Vector store reset at: {CHROMA_DIR}")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise


def get_vector_store() -> Chroma:
    """Open the persistent Chroma collection."""

    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Opening Chroma collection: {COLLECTION_NAME}")
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embedding_model(),
            client_settings=Settings(anonymized_telemetry=False),
        )
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise


def store_chunks(chunks: list[Document]) -> None:
    """Save chunks and their embeddings in Chroma."""

    valid_chunks = _validate_chunks(chunks)
    print(f"Vector store received {len(valid_chunks)} valid chunks")
    print(f"Vector backend selected: {'SQLite fallback' if _use_simple_store() else 'ChromaDB'}")

    if _use_simple_store():
        _store_chunks_sqlite(valid_chunks)
        return

    try:
        print("Storing in ChromaDB...")
        vector_store = get_vector_store()
        vector_store.add_documents(valid_chunks)
        print("ChromaDB insertion completed")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Could not create embeddings or save them in Chroma: {exc}") from exc


def retrieve_relevant_chunks(question: str, k: int = 6) -> list[Document]:
    """Return top matching chunks for a user question."""

    if _use_simple_store():
        return _retrieve_chunks_sqlite(question, k=k)

    try:
        print("Searching ChromaDB...")
        vector_store = get_vector_store()
        results = vector_store.max_marginal_relevance_search(
    question,
    k=k,
    fetch_k=12,
)
        print(f"Retrieved {len(results)} chunks from ChromaDB")
        return results
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Could not search the vector database: {exc}") from exc


def _validate_chunks(chunks: list[Document]) -> list[Document]:
    """Remove blank chunks and sanitize metadata before embeddings."""

    valid_chunks: list[Document] = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.page_content.strip()
        if not text:
            continue

        chunk.page_content = text
        chunk.metadata = _sanitize_metadata(chunk.metadata, fallback_chunk_id=f"chunk-{index}")
        valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError("No valid text chunks generated")

    return valid_chunks


def _sanitize_metadata(metadata: dict, fallback_chunk_id: str) -> dict[str, str | int | float | bool]:
    """Ensure metadata only contains Chroma-supported primitive values."""

    source = metadata.get("source") or "unknown"
    page = metadata.get("page", 0)
    chunk_id = metadata.get("chunk_id") or fallback_chunk_id

    try:
        page_number = int(page) if page is not None else 0
    except (TypeError, ValueError):
        page_number = 0

    return {
        "source": str(source),
        "page": page_number,
        "chunk_id": str(chunk_id),
    }


def _get_sqlite_connection(db_path: Path = SIMPLE_STORE_PATH) -> sqlite3.Connection:
    """Open the lightweight local vector store."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    return connection


def _store_chunks_sqlite(chunks: list[Document]) -> None:
    """Embed and store chunks in a simple SQLite table."""

    try:
        print("Generating embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embed_texts(texts)
        print(f"Generated {len(embeddings)} embeddings")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Local embedding generation failed: {exc}") from exc

    rows = [
        (
            chunk.page_content,
            json.dumps(chunk.metadata),
            json.dumps(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]

    try:
        print(f"Storing embeddings in SQLite vector fallback: {SIMPLE_STORE_PATH}")
        with _get_sqlite_connection() as connection:
            connection.executemany(
                "INSERT INTO chunks (content, metadata, embedding) VALUES (?, ?, ?)",
                rows,
            )
            connection.commit()
        print("SQLite vector fallback insertion completed")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Could not store embeddings in SQLite fallback: {exc}") from exc


def _retrieve_chunks_sqlite(question: str, k: int = 4) -> list[Document]:
    """Retrieve top chunks with cosine similarity in the SQLite fallback."""

    try:
        print("Generating query embedding...")
        query_embedding = embed_query(question)
        print("Query embedding generated")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Local query embedding failed: {exc}") from exc

    try:
        with _get_sqlite_connection() as connection:
            rows = connection.execute("SELECT content, metadata, embedding FROM chunks").fetchall()
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Could not read SQLite vector fallback: {exc}") from exc

    scored: list[tuple[float, Document]] = []
    for content, metadata_json, embedding_json in rows:
        embedding = json.loads(embedding_json)
        score = _cosine_similarity(query_embedding, embedding)
        scored.append(
            (
                score,
                Document(
                    page_content=content,
                    metadata=json.loads(metadata_json),
                ),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    print(f"Retrieved {min(len(scored), k)} chunks from SQLite vector fallback")
    return [document for _, document in scored[:k]]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)
