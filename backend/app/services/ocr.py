"""OCR service for extracting text from images."""


from pathlib import Path

import pytesseract
from PIL import Image

from langchain_core.documents import Document

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


class OCRProcessingError(Exception):
    """Raised when OCR extraction fails."""


def extract_text_from_image(file_path: Path) -> list[Document]:
    """Extract text from an uploaded image."""

    try:

        image = Image.open(file_path).convert("L")

        extracted_text = pytesseract.image_to_string(image)

        extracted_text = extracted_text.strip()

        if not extracted_text:
            raise OCRProcessingError(
                "No readable text found in image."
            )

        return [
            Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path.name,
                    "page": 1,
                },
            )
        ]

    except Exception as exc:

        raise OCRProcessingError(
            f"OCR extraction failed: {exc}"
        ) from exc