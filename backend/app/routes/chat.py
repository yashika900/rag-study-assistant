"""Chat endpoints."""

import traceback

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from backend.app.db.database import add_chat_message, clear_chat_history
from backend.app.models.schemas import ChatRequest, ChatResponse, ClearChatResponse
from backend.app.services.rag_pipeline import answer_question


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer a user question from the uploaded study material."""

    print("\n===== CHAT PIPELINE START =====")
    question = request.question.strip()
    print(f"Question received: {question}")
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        answer, sources = answer_question(question)
        print(f"Answer generated. Source count: {len(sources)}")
        add_chat_message(question, answer)
        print("Chat history saved")
        print("===== CHAT PIPELINE END =====\n")
        return ChatResponse(answer=answer, sources=sources)
    except RuntimeError as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(exc)},
        )


@router.post("/clear-chat", response_model=ClearChatResponse)
async def clear_chat() -> ClearChatResponse:
    """Clear stored chat history for the current local session."""

    print("Clearing chat history...")
    clear_chat_history()
    return ClearChatResponse(message="Chat history cleared.")
