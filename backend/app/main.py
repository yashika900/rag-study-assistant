"""FastAPI entry point for the RAG Study Assistant backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.db.database import init_db
from backend.app.routes import chat, upload
from backend.app.routes.export import router as export_router
from backend.app.utils.helpers import ensure_project_dirs


app = FastAPI(
    title="RAG Study Assistant API",
    description="Upload study material and ask grounded questions from it.",
    version="0.1.0",
)

# ── Single CORS middleware (was accidentally added twice before) ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(export_router)
app.include_router(upload.router)
app.include_router(chat.router)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    """Prepare runtime folders and SQLite tables."""
    ensure_project_dirs()
    init_db()


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
async def health_check() -> dict[str, str]:
    """Simple health endpoint for local checks and Render."""
    return {"status": "ok", "message": "RAG Study Assistant API is running."}