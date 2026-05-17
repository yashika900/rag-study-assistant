"""Pydantic request and response models used by the API."""

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """A chunk returned to the UI as a citation."""

    content: str
    source: str
    page: int | None = None
    chunk_id: str | None = None


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class ClearChatResponse(BaseModel):
    message: str
