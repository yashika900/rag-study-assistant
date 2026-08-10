# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies (Tesseract OCR + clean apt cache) ────────────────────
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY backend/ ./backend/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 10000

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "10000"]