"""Small helper functions shared across the backend."""

from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
SQLITE_PATH = BASE_DIR / "backend" / "study_assistant.db"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
}


def ensure_project_dirs() -> None:
    """Create folders that the app writes to at runtime."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """Return a normalized file extension such as .pdf."""

    return Path(filename).suffix.lower()


def is_supported_file(filename: str) -> bool:
    """Check whether the file type is supported by the parser service."""

    return get_file_extension(filename) in SUPPORTED_EXTENSIONS


def safe_upload_filename(filename: str) -> str:
    """Create a filesystem-safe name while keeping the original extension."""

    original = Path(filename)
    stem = "".join(char for char in original.stem if char.isalnum() or char in {"-", "_"}).strip()
    clean_stem = stem or "document"
    return f"{clean_stem}_{uuid4().hex[:8]}{original.suffix.lower()}"
