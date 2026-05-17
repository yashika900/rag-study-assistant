"""Text chunking service."""
import uuid
import traceback

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for better retrieval."""

    print(f"Chunker received {len(documents)} documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        raw_chunks = splitter.split_documents(documents)
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise

    chunks: list[Document] = []
    for index, chunk in enumerate(raw_chunks, start=1):
        text = chunk.page_content.strip()
        if not text:
            continue

        chunk.page_content = text
        chunk.metadata = _sanitize_metadata(
            chunk.metadata,
            chunk_id=f"chunk-{uuid.uuid4()}"
            )
        chunks.append(chunk)

    print(f"Chunker generated {len(chunks)} valid chunks")
    if not chunks:
        raise ValueError("No valid text chunks generated")

    return chunks


def _sanitize_metadata(metadata: dict, chunk_id: str) -> dict[str, str | int | float | bool]:
    """Keep metadata simple enough for ChromaDB."""

    source = metadata.get("source") or "unknown"
    page = metadata.get("page", 0)

    try:
        page_number = int(page) if page is not None else 0
    except (TypeError, ValueError):
        page_number = 0

    return {
        "source": str(source),
        "page": page_number,
        "chunk_id": str(chunk_id),
    }
