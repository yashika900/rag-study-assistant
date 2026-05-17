"""PDF export service."""

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


def generate_chat_pdf(chat_history: list[dict]) -> BytesIO:
    """Generate PDF from chat history."""

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "RAG Study Assistant Chat Export",
        styles["Title"],
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    for chat in chat_history:

        question = chat.get("question", "")

        answer = chat.get("answer", "")

        question_paragraph = Paragraph(
            f"<b>Question:</b><br/>{question}",
            styles["BodyText"],
        )

        answer_paragraph = Paragraph(
            f"<b>Answer:</b><br/>{answer}",
            styles["BodyText"],
        )

        elements.append(question_paragraph)

        elements.append(Spacer(1, 10))

        elements.append(answer_paragraph)

        elements.append(Spacer(1, 20))

    doc.build(elements)

    buffer.seek(0)

    return buffer