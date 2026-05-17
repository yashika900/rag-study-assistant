"""RAG pipeline: retrieve chunks, build a grounded prompt, ask Gemini."""

import traceback

from langchain_core.documents import Document

from backend.app.models.schemas import SourceChunk
from backend.app.services.llm import invoke_llm_with_fallback
from backend.app.services.vector_store import retrieve_relevant_chunks


NOT_FOUND_MESSAGE = "I could not find this in the uploaded material."


def answer_question(question: str) -> tuple[str, list[SourceChunk]]:
    """Answer a question using only retrieved document context."""

    try:
        print("Retrieving relevant chunks...")
        chunks = retrieve_relevant_chunks(question)
        print(f"Retrieved {len(chunks)} relevant chunks")
        if not chunks:
            return NOT_FOUND_MESSAGE, []

        context = _format_context(chunks)
        print(f"Context length: {len(context)}")
        prompt = f"""
You are an intelligent AI Study Assistant.

Answer the question ONLY using the provided context.

Instructions:
- Give complete and detailed answers.
- If multiple concepts or methods are mentioned in the context, include ALL of them.
- Organize answers using bullet points or sections when appropriate.
- Explain concepts clearly for a student.
- Include important details, definitions, steps, and examples whenever available.
- Do NOT ignore relevant information from other retrieved chunks.
- If the answer is not present in the context, reply exactly:
"{NOT_FOUND_MESSAGE}"

Context:
{context}

Question:
{question}

Detailed Answer:
""".strip()
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise

    try:
        print("STEP 1: Sending prompt to Gemini...")
        answer = invoke_llm_with_fallback(prompt)
        print("STEP 2: Gemini response received")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Gemini could not generate an answer right now: {exc}") from exc

    sources = [_source_from_document(chunk) for chunk in chunks]
    return answer or NOT_FOUND_MESSAGE, sources


def _format_context(chunks: list[Document]) -> str:
    """Format chunks with source-aware citations."""

    formatted_chunks = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown source")
        page = chunk.metadata.get("page")

        page_label = f", page {page}" if page else ""

        formatted_chunks.append(
            f"""
[Document {index}]
Source: {source}{page_label}
Content:
{chunk.page_content}
""".strip()
        )

    return "\n\n".join(formatted_chunks)
    """Format chunks with citation labels so the model can refer to sources."""

    formatted_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown source")
        page = chunk.metadata.get("page")
        page_label = f", page {page}" if page else ""
        formatted_chunks.append(
            f"[Source {index}: {source}{page_label}]\n{chunk.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def _source_from_document(document: Document) -> SourceChunk:
    """Convert a LangChain document into an API citation model."""

    page = document.metadata.get("page")
    return SourceChunk(
        content=document.page_content,
        source=str(document.metadata.get("source", "unknown source")),
        page=int(page) if isinstance(page, int) else None,
        chunk_id=document.metadata.get("chunk_id"),
    )
