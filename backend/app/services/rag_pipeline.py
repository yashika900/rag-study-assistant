"""RAG pipeline: detect study intent, retrieve context, and ask Gemini."""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass

from langchain_core.documents import Document

from backend.app.db.database import get_chat_history
from backend.app.models.schemas import SourceChunk
from backend.app.services.llm import invoke_llm_with_fallback
from backend.app.services.vector_store import retrieve_all_chunks, retrieve_relevant_chunks


NOT_FOUND_MESSAGE = (
    "I could not find this in the uploaded material. I can only answer from the "
    "documents you indexed in this session, so please upload the relevant notes "
    "if you want me to cover that topic."
)


@dataclass(frozen=True)
class IntentConfig:
    """Retrieval and prompt settings for one study task type."""

    name: str
    retrieval_k: int
    fetch_k: int
    max_context_chars: int
    use_all_chunks: bool = False
    all_chunk_limit: int = 120


INTENT_CONFIGS: dict[str, IntentConfig] = {
    "summary": IntentConfig("summary", 18, 45, 15000, use_all_chunks=True),
    "topic_listing": IntentConfig("topic_listing", 14, 40, 12000, use_all_chunks=True),
    "definition": IntentConfig("definition", 4, 12, 5000),
    "comparison": IntentConfig("comparison", 8, 24, 9000),
    "detailed_explanation": IntentConfig("detailed_explanation", 18, 45, 16000),
    "study_notes": IntentConfig("study_notes", 18, 45, 16000),
    "interview_questions": IntentConfig("interview_questions", 20, 50, 17000, use_all_chunks=True),
    "mcq": IntentConfig("mcq", 20, 50, 17000, use_all_chunks=True),
    "flashcards": IntentConfig("flashcards", 18, 45, 15000, use_all_chunks=True),
    "exam_prep": IntentConfig("exam_prep", 20, 50, 17000, use_all_chunks=True),
    "page_lookup": IntentConfig("page_lookup", 6, 18, 7000),
    "procedural": IntentConfig("procedural", 8, 24, 9000),
    "general_qa": IntentConfig("general_qa", 6, 18, 8000),
}


def answer_question(question: str) -> tuple[str, list[SourceChunk]]:
    """Answer a question using only indexed study material."""

    try:
        intent = detect_intent(question)
        config = INTENT_CONFIGS[intent]
        print(f"Detected study intent: {intent}")

        retrieval_query = _build_retrieval_query(question)
        print(f"Retrieval query: {retrieval_query}")

        if config.use_all_chunks:
            print(f"Loading broad context. Limit: {config.all_chunk_limit}")
            chunks = retrieve_all_chunks(limit=config.all_chunk_limit)
        else:
            print(
                "Retrieving relevant chunks "
                f"(k={config.retrieval_k}, fetch_k={config.fetch_k})..."
            )
            chunks = retrieve_relevant_chunks(
                retrieval_query,
                k=config.retrieval_k,
                fetch_k=config.fetch_k,
            )

        chunks = _prepare_chunks_for_prompt(chunks, config)
        print(f"Prepared {len(chunks)} chunks for prompt")

        if not chunks:
            return NOT_FOUND_MESSAGE, []

        context = _format_context(chunks, max_chars=config.max_context_chars)
        uploaded_sources = sorted(
            {str(chunk.metadata.get("source", "unknown")) for chunk in chunks}
        )
        print(f"Context length: {len(context)}")
        print(f"Context sources: {', '.join(uploaded_sources)}")

        prompt = _build_prompt(
            question=question,
            context=context,
            uploaded_sources=uploaded_sources,
            intent=intent,
        )
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise

    try:
        print("Sending prompt to Gemini...")
        answer = invoke_llm_with_fallback(prompt)
        print("Gemini response received")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise RuntimeError(f"Gemini could not generate an answer right now: {exc}") from exc

    if _looks_like_not_found(answer):
        return answer.strip() or NOT_FOUND_MESSAGE, []

    sources = [_source_from_document(chunk) for chunk in chunks]
    return _clean_answer(answer) or NOT_FOUND_MESSAGE, sources


def detect_intent(question: str) -> str:
    """Classify the study task using lightweight keyword rules."""

    text = _normalize(question)

    if _contains_any(text, ("mcq", "multiple choice", "quiz questions", "objective questions")):
        return "mcq"
    if _contains_any(text, ("flashcard", "flash card", "flashcards")):
        return "flashcards"
    if _contains_any(text, ("interview question", "viva", "oral exam")):
        return "interview_questions"
    if _contains_any(text, ("exam prep", "important questions", "exam point", "quick revision")):
        return "exam_prep"
    if _contains_any(text, ("study notes", "make notes", "revision notes", "short notes")):
        return "study_notes"
    if _contains_any(text, ("compare", "difference between", "versus", " vs ", "distinguish")):
        return "comparison"
    if _contains_any(text, ("list topics", "topic list", "headings", "main topics", "chapters")):
        return "topic_listing"
    if _contains_any(text, ("summarize", "summary", "overview", "briefly cover")):
        return "summary"
    if _contains_any(text, ("define", "definition", "what is", "what are", "meaning of")):
        return "definition"
    if _contains_any(text, ("explain", "describe", "elaborate", "in detail", "teach me")):
        return "detailed_explanation"
    if _contains_any(text, ("steps", "procedure", "process", "how does", "how to")):
        return "procedural"
    if _contains_any(text, ("page", "where is", "which page")):
        return "page_lookup"

    return "general_qa"


def _build_retrieval_query(question: str) -> str:
    """Expand the search query without making another LLM call."""

    expanded_parts = [question.strip()]
    normalized = _normalize(question)

    acronym_expansions = {
        "osi": "open systems interconnection layers model",
        "tcp": "transmission control protocol reliable connection oriented",
        "udp": "user datagram protocol unreliable connectionless",
        "ip": "internet protocol addressing routing",
        "lan": "local area network",
        "wan": "wide area network",
        "man": "metropolitan area network",
        "pan": "personal area network",
        "dns": "domain name system",
        "http": "hypertext transfer protocol",
        "https": "secure hypertext transfer protocol",
        "mac": "media access control address",
        "arp": "address resolution protocol",
    }
    for acronym, expansion in acronym_expansions.items():
        if re.search(rf"\b{re.escape(acronym)}\b", normalized):
            expanded_parts.append(expansion)

    synonym_groups = {
        "summarize": "summary overview key points main ideas",
        "advantages": "benefits uses importance",
        "disadvantages": "limitations drawbacks problems",
        "types": "classification categories kinds",
        "function": "purpose role working",
    }
    for trigger, expansion in synonym_groups.items():
        if trigger in normalized:
            expanded_parts.append(expansion)

    # Follow-up questions like "explain it again" need the previous user topic.
    if _is_follow_up(normalized):
        recent_questions = [
            item.get("question", "")
            for item in get_chat_history(limit=3)
            if item.get("question")
        ]
        if recent_questions:
            expanded_parts.insert(0, " ".join(recent_questions))

    return " ".join(part for part in expanded_parts if part).strip()


def _prepare_chunks_for_prompt(
    chunks: list[Document],
    config: IntentConfig,
) -> list[Document]:
    """Dedupe, diversify, and lightly merge chunks before prompting."""

    cleaned = _dedupe_chunks(chunks)
    if config.use_all_chunks:
        cleaned = _interleave_chunks_by_source(cleaned)
    else:
        cleaned = _diversify_chunks(cleaned, limit=config.retrieval_k)
    merged = _merge_neighboring_chunks(cleaned)
    return _limit_chunks_by_chars(merged, max_chars=config.max_context_chars)


def _dedupe_chunks(chunks: list[Document]) -> list[Document]:
    """Remove repeated or near-identical chunks."""

    seen: set[str] = set()
    unique_chunks: list[Document] = []
    for chunk in chunks:
        normalized = " ".join(chunk.page_content.lower().split())
        key = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page')}|{normalized[:240]}"
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append(chunk)
    return unique_chunks


def _diversify_chunks(chunks: list[Document], limit: int) -> list[Document]:
    """Prefer a mix of sources/pages while preserving retrieval order."""

    selected: list[Document] = []
    counts_by_source: dict[str, int] = {}
    max_per_source_first_pass = max(2, limit // 2)

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        if counts_by_source.get(source, 0) >= max_per_source_first_pass:
            continue
        selected.append(chunk)
        counts_by_source[source] = counts_by_source.get(source, 0) + 1
        if len(selected) >= limit:
            return selected

    for chunk in chunks:
        if chunk not in selected:
            selected.append(chunk)
        if len(selected) >= limit:
            break

    return selected


def _interleave_chunks_by_source(chunks: list[Document]) -> list[Document]:
    """Balance broad prompts so each uploaded document contributes context."""

    buckets: dict[str, list[Document]] = {}
    source_order: list[str] = []

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        if source not in buckets:
            buckets[source] = []
            source_order.append(source)
        buckets[source].append(chunk)

    interleaved: list[Document] = []
    while any(buckets.values()):
        for source in source_order:
            if buckets[source]:
                interleaved.append(buckets[source].pop(0))

    return interleaved


def _merge_neighboring_chunks(chunks: list[Document]) -> list[Document]:
    """Merge chunks from the same source/page so explanations read naturally."""

    merged: list[Document] = []
    grouped: dict[tuple[str, int], list[Document]] = {}

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = int(chunk.metadata.get("page") or 0)
        grouped.setdefault((source, page), []).append(chunk)

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = int(chunk.metadata.get("page") or 0)
        key = (source, page)
        group = grouped.pop(key, None)
        if not group:
            continue

        text_parts: list[str] = []
        for item in group:
            if item.page_content not in text_parts:
                text_parts.append(item.page_content)
        merged_text = "\n".join(text_parts).strip()
        metadata = dict(group[0].metadata)
        merged.append(Document(page_content=merged_text, metadata=metadata))

    return merged


def _limit_chunks_by_chars(chunks: list[Document], max_chars: int) -> list[Document]:
    """Keep prompt size predictable for Gemini."""

    selected: list[Document] = []
    total_chars = 0
    for chunk in chunks:
        chunk_len = len(chunk.page_content)
        if selected and total_chars + chunk_len > max_chars:
            break
        selected.append(chunk)
        total_chars += chunk_len
    return selected


def _format_context(chunks: list[Document], max_chars: int) -> str:
    """Format chunks with compact source-aware citations."""

    formatted_chunks: list[str] = []
    total_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown source")
        page = chunk.metadata.get("page")
        page_label = f", page {page}" if page else ""
        block = (
            f"[Source {index}]\n"
            f"Source: {source}{page_label}\n"
            f"Content:\n{chunk.page_content}"
        )
        if formatted_chunks and total_chars + len(block) > max_chars:
            break
        formatted_chunks.append(block)
        total_chars += len(block)

    return "\n\n".join(formatted_chunks)


def _build_prompt(
    question: str,
    context: str,
    uploaded_sources: list[str],
    intent: str,
) -> str:
    """Build an intent-specific prompt for Gemini."""

    base_rules = f"""
You are a helpful Study Assistant.

Use ONLY the context below from the currently uploaded/indexed documents:
{", ".join(uploaded_sources)}

Grounding rules:
- Do not use outside knowledge.
- Do not mention documents that are not listed above.
- If the answer is not supported by the context, say that it is not in the uploaded material in 2-4 sentences.
- Cite page/source details naturally when useful.
- Avoid repeating the same point.
""".strip()

    task_rules = {
        "summary": """
Task: Create a concise summary.
- If multiple documents are present, summarize each document separately.
- Use short bullets with the key ideas only.
- End with 3-5 quick takeaways.
""",
        "topic_listing": """
Task: List the topics/headings present in the material.
- Return topics only, grouped by source if possible.
- Do not explain each topic unless the question asks for explanation.
""",
        "definition": """
Task: Give a clear definition.
- Start directly with the definition.
- Add key points and examples only if the context includes them.
- Keep it concise and student-friendly.
""",
        "comparison": """
Task: Compare the requested concepts.
- Use a Markdown table.
- Add a short conclusion after the table.
- Only compare points found in the context.
""",
        "detailed_explanation": """
Task: Explain the concept clearly for a student.
- Use headings and bullets.
- Include working, examples, steps, and important exam points when present.
- Make the answer complete but not padded.
""",
        "study_notes": """
Task: Create study notes.
- Use compact headings, bullets, key terms, and quick revision points.
- Highlight exam-relevant ideas only when supported by the context.
""",
        "interview_questions": """
Task: Generate interview/viva questions from the material.
- Provide numbered questions with short answers.
- Keep every question grounded in the context.
""",
        "mcq": """
Task: Generate MCQs from the material.
- Use four options: A, B, C, D.
- Include the correct answer after each MCQ.
- Do not invent facts outside the context.
""",
        "flashcards": """
Task: Create flashcards.
- Use a two-column Markdown table: Question | Answer.
- Keep cards short and useful for revision.
""",
        "exam_prep": """
Task: Prepare exam-focused revision help.
- Include important topics, likely questions, and quick revision bullets.
- Only include points supported by the uploaded material.
""",
        "page_lookup": """
Task: Locate information in the uploaded material.
- Mention source names and page numbers when present.
- Briefly state what is found there.
""",
        "procedural": """
Task: Explain the process or steps.
- Use a numbered list when the context describes a sequence.
- Include conditions or examples only when present.
""",
        "general_qa": """
Task: Answer the question directly.
- Use concise paragraphs or bullets, whichever fits best.
- Include citations/source references when helpful.
""",
    }

    return f"""
{base_rules}

{task_rules.get(intent, task_rules["general_qa"]).strip()}

Context:
{context}

Question:
{question}

Answer:
""".strip()


def _looks_like_not_found(answer: str) -> bool:
    """Detect grounded refusal answers so the UI does not show misleading citations."""

    normalized = _normalize(answer)
    return (
        "could not find" in normalized
        or "not in the uploaded material" in normalized
        or "not supported by the context" in normalized
    )


def _clean_answer(answer: str) -> str:
    """Remove accidental duplicated whitespace from the model answer."""

    return re.sub(r"\n{3,}", "\n\n", answer.strip())


def _is_follow_up(normalized_question: str) -> bool:
    """Detect questions that likely refer to the previous turn."""

    follow_up_terms = (
        "it",
        "this",
        "that",
        "these",
        "those",
        "again",
        "more",
        "in detail",
        "continue",
    )
    return len(normalized_question.split()) <= 8 and _contains_any(
        normalized_question,
        follow_up_terms,
    )


def _normalize(text: str) -> str:
    """Lowercase text and normalize spacing."""

    return f" {' '.join(text.lower().split())} "


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return True when any phrase appears in normalized text."""

    return any(term in text for term in terms)


def _source_from_document(document: Document) -> SourceChunk:
    """Convert a LangChain document into an API citation model."""

    page = document.metadata.get("page")
    return SourceChunk(
        content=document.page_content,
        source=str(document.metadata.get("source", "unknown source")),
        page=int(page) if isinstance(page, int) else None,
        chunk_id=document.metadata.get("chunk_id"),
    )
