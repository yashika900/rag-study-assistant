"""Vector storage service.

Chroma is the intended backend. On Windows + Python 3.12, current Chroma wheels
can crash in native code during inserts, so this module includes a tiny SQLite
fallback that keeps the MVP usable while preserving the same service interface.

Session isolation: every upload batch is tagged with a session_id so that
retrieve_all_chunks() and retrieve_relevant_chunks() never return stale data
from a previous app session or a previous upload batch.
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
from uuid import uuid4

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.app.services.embeddings import embed_query, embed_texts, get_embedding_model
from backend.app.utils.helpers import CHROMA_DIR


COLLECTION_NAME = "study_material"
SIMPLE_STORE_PATH = CHROMA_DIR / "simple_vector_store.sqlite"

# ---------------------------------------------------------------------------
# Session state - lives in memory for the duration of the process.
# Reset whenever the user uploads a new batch of documents.
# ---------------------------------------------------------------------------
_current_session_id: str = str(uuid4())


def get_current_session_id() -> str:
    """Return the active session ID."""
    return _current_session_id


def new_session() -> str:
    """Generate and store a fresh session ID. Call this on every upload reset."""
    global _current_session_id
    _current_session_id = str(uuid4())
    print(f"New session started: {_current_session_id}")
    return _current_session_id


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

def _use_simple_store() -> bool:
    """Choose the local SQLite fallback when Chroma is unsafe locally."""
    backend = os.getenv("VECTOR_STORE_BACKEND", "").strip().lower()
    if backend == "chroma":
        return False
    if backend == "simple":
        return True
    return platform.system() == "Windows" and sys.version_info >= (3, 12)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reset_vector_store() -> None:
    """Remove existing embeddings and start a new session."""
    try:
        new_session()  # New session ID on every reset.

        if _use_simple_store():
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            with _get_sqlite_connection() as connection:
                connection.execute("DELETE FROM chunks")
                connection.commit()
            print(f"SQLite vector fallback cleared at: {SIMPLE_STORE_PATH}")
            return

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
    """Save chunks and their embeddings, tagged with the current session ID."""
    valid_chunks = _validate_chunks(chunks)
    print(f"Vector store received {len(valid_chunks)} valid chunks")
    print(f"Session ID: {_current_session_id}")
    print(f"Vector backend: {'SQLite fallback' if _use_simple_store() else 'ChromaDB'}")

    if _use_simple_store():
        _store_chunks_sqlite(valid_chunks)
        return

    try:
        print("Storing in ChromaDB...")
        # Tag every chunk with the current session ID
        for chunk in valid_chunks:
            chunk.metadata["session_id"] = _current_session_id
        vector_store = get_vector_store()
        vector_store.add_documents(valid_chunks)
        print("ChromaDB insertion completed")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(
            f"Could not create embeddings or save them in Chroma: {exc}"
        ) from exc


def retrieve_relevant_chunks(question: str, k: int = 6, fetch_k: int | None = None) -> list[Document]:
    """Return top matching chunks for the current session only."""
    candidate_count = fetch_k or max(12, k * 3)
    if _use_simple_store():
        return _retrieve_chunks_sqlite(question, k=k, fetch_k=candidate_count)

    try:
        print(f"Searching ChromaDB (session: {_current_session_id})...")
        vector_store = get_vector_store()
        results = vector_store.max_marginal_relevance_search(
            question,
            k=k,
            fetch_k=candidate_count,
            filter={"session_id": _current_session_id},
        )
        print(f"Retrieved {len(results)} chunks from ChromaDB")
        return results
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Could not search the vector database: {exc}") from exc


def retrieve_all_chunks(limit: int = 120) -> list[Document]:
    """Return all chunks for the current session (used for summarization)."""
    if _use_simple_store():
        return _retrieve_all_chunks_sqlite(limit=limit)

    try:
        print(f"Reading all chunks from ChromaDB (session: {_current_session_id})...")
        vector_store = get_vector_store()
        raw = vector_store.get(
            include=["documents", "metadatas"],
            limit=limit,
            where={"session_id": _current_session_id},
        )
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        chunks = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas, strict=False)
            if doc
        ]
        print(f"Loaded {len(chunks)} chunks from ChromaDB")
        return chunks
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(
            f"Could not read all chunks from vector database: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_chunks(chunks: list[Document]) -> list[Document]:
    """Remove blank chunks and sanitize metadata before embeddings."""
    valid_chunks: list[Document] = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.page_content.strip()
        if not text:
            continue
        chunk.page_content = text
        chunk.metadata = _sanitize_metadata(
            chunk.metadata, fallback_chunk_id=f"chunk-{index}"
        )
        valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError("No valid text chunks generated")
    return valid_chunks


def _sanitize_metadata(
    metadata: dict, fallback_chunk_id: str
) -> dict[str, str | int | float | bool]:
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
        "session_id": _current_session_id,  # Always tag metadata.
    }


# ---------------------------------------------------------------------------
# SQLite fallback - all queries scoped to current session
# ---------------------------------------------------------------------------

def _get_sqlite_connection(db_path: Path = SIMPLE_STORE_PATH) -> sqlite3.Connection:
    """Open the lightweight local vector store, adding session_id column if needed."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    # Create table with session_id column
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT    NOT NULL,
            metadata    TEXT    NOT NULL,
            embedding   TEXT    NOT NULL,
            session_id  TEXT    NOT NULL DEFAULT ''
        )
        """
    )
    # Add session_id column to existing tables that predate this change
    try:
        connection.execute("ALTER TABLE chunks ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
        connection.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists.
    return connection


def _store_chunks_sqlite(chunks: list[Document]) -> None:
    """Embed and store chunks in SQLite, tagged with the current session ID."""
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
            _current_session_id,      # Tag each row.
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]

    try:
        print(f"Storing embeddings in SQLite: {SIMPLE_STORE_PATH}")
        with _get_sqlite_connection() as connection:
            connection.executemany(
                "INSERT INTO chunks (content, metadata, embedding, session_id) VALUES (?, ?, ?, ?)",
                rows,
            )
            connection.commit()
        print("SQLite insertion completed")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(
            f"Could not store embeddings in SQLite fallback: {exc}"
        ) from exc


def _retrieve_chunks_sqlite(question: str, k: int = 4, fetch_k: int | None = None) -> list[Document]:
    """Retrieve top chunks for the current session using cosine similarity."""
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
            # Only fetch rows belonging to the current session
            rows = connection.execute(
                "SELECT content, metadata, embedding FROM chunks WHERE session_id = ?",
                (_current_session_id,),
            ).fetchall()
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Could not read SQLite vector fallback: {exc}") from exc

    print(f"Scoring {len(rows)} chunks for current session")
    scored: list[tuple[float, Document]] = []
    for content, metadata_json, embedding_json in rows:
        embedding = json.loads(embedding_json)
        score = _cosine_similarity(query_embedding, embedding)
        scored.append(
            (score, Document(page_content=content, metadata=json.loads(metadata_json)))
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    candidates = scored[: fetch_k or max(k * 3, k)]
    top = _select_diverse_documents(candidates, k=k)
    print(f"Retrieved {len(top)} chunks from SQLite")
    return top


def _retrieve_all_chunks_sqlite(limit: int = 120) -> list[Document]:
    """Read all chunks for the current session in insertion order."""
    try:
        with _get_sqlite_connection() as connection:
            rows = connection.execute(
                """
                SELECT content, metadata
                FROM   chunks
                WHERE  session_id = ?
                ORDER  BY id ASC
                LIMIT  ?
                """,
                (_current_session_id, limit),
            ).fetchall()
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Could not read all SQLite vector chunks: {exc}") from exc

    chunks = [
        Document(page_content=content, metadata=json.loads(meta))
        for content, meta in rows
    ]
    print(f"Loaded {len(chunks)} chunks from current session")
    return chunks


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _select_diverse_documents(
    scored_documents: list[tuple[float, Document]],
    k: int,
) -> list[Document]:
    """Select relevant chunks while avoiding too many from one source/page."""

    selected: list[Document] = []
    seen_pages: set[tuple[str, int]] = set()
    max_per_page_first_pass = 2

    for _, document in scored_documents:
        source = str(document.metadata.get("source", "unknown"))
        page = int(document.metadata.get("page") or 0)
        page_key = (source, page)
        page_count = sum(
            1
            for selected_doc in selected
            if (
                str(selected_doc.metadata.get("source", "unknown")),
                int(selected_doc.metadata.get("page") or 0),
            )
            == page_key
        )
        if page_key in seen_pages and page_count >= max_per_page_first_pass:
            continue
        selected.append(document)
        seen_pages.add(page_key)
        if len(selected) >= k:
            return selected

    for _, document in scored_documents:
        if document not in selected:
            selected.append(document)
        if len(selected) >= k:
            break

    return selected
