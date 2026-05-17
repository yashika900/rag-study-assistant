"""PDF export endpoint."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.app.db.database import get_chat_history
from backend.app.services.pdf_export import generate_chat_pdf


router = APIRouter()


@router.get("/export-pdf")
async def export_chat_pdf():
    """Export current chat history as PDF."""

    messages = get_chat_history()

    pdf_buffer = generate_chat_pdf(messages)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                "attachment; filename=study_assistant_chat.pdf"
            )
        },
    )