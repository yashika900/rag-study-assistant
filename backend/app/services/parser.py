"""Document parsing service for PDF, TXT, and DOCX files."""

from backend.app.services.ocr import (
    extract_text_from_image,
    SUPPORTED_IMAGE_TYPES,
)
import traceback
from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document
from pypdf import PdfReader


class DocumentParsingError(Exception):
    """Raised when a file cannot be parsed into usable text."""


def parse_document(file_path: Path) -> list[Document]:
    """Extract text and basic metadata from a supported document."""

    print(f"Parser started for: {file_path}")
    extension = file_path.suffix.lower()
    try:
        if extension == ".pdf":
            documents = _parse_pdf(file_path)
        elif extension == ".txt":
            documents = _parse_txt(file_path)
        elif extension == ".docx":
            documents = _parse_docx(file_path)
        elif extension in SUPPORTED_IMAGE_TYPES:
            documents = extract_text_from_image(file_path)
        else:
            raise DocumentParsingError(f"Unsupported file type: {extension}")
    except DocumentParsingError:
        raise
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise DocumentParsingError(str(exc)) from exc

    if not any(doc.page_content.strip() for doc in documents):
        raise DocumentParsingError("The uploaded file does not contain readable text.")

    total_length = sum(len(document.page_content) for document in documents)
    print(f"Parser complete. Documents: {len(documents)}, text length: {total_length}")
    return documents


def _parse_pdf(file_path: Path) -> list[Document]:
    """Extract one LangChain document per PDF page."""

    try:
        reader = PdfReader(str(file_path))
        print(f"PDF page count: {len(reader.pages)}")
        documents: list[Document] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=_clean_text(text),
                        metadata={"source": file_path.name, "page": index},
                    )
                )
        return documents
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise DocumentParsingError("Could not read the PDF file.") from exc


def _parse_txt(file_path: Path) -> list[Document]:
    """Read a plain text file as a single document."""

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise DocumentParsingError("Could not read the text file.") from exc

    cleaned_text = _clean_text(text)
    if not cleaned_text:
        raise DocumentParsingError("Could not extract text from TXT")

    return [
        Document(
            page_content=cleaned_text,
            metadata={"source": file_path.name, "page": 0},
        )
    ]


def _parse_docx(file_path: Path) -> list[Document]:
    """Extract paragraph text from a DOCX file."""

    try:
        docx = DocxDocument(str(file_path))
        text = "\n".join(paragraph.text for paragraph in docx.paragraphs)
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise DocumentParsingError("Could not read the DOCX file.") from exc

    cleaned_text = _clean_text(text)
    if not cleaned_text:
        raise DocumentParsingError("Could not extract text from DOCX")

    return [
        Document(
            page_content=cleaned_text,
            metadata={"source": file_path.name, "page": 0},
        )
    ]


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""

    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
