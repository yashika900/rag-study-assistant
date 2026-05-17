# RAG Study Assistant

A beginner-to-intermediate Retrieval Augmented Generation project that lets students upload study material and ask questions grounded only in the uploaded content.

## Features

- Upload PDF, TXT, and DOCX files
- Extract readable text with page metadata for PDFs
- Split text into overlapping LangChain chunks
- Generate local sentence-transformers embeddings and store them in ChromaDB
- Retrieve relevant chunks for each question
- Ask Gemini to answer only from retrieved context
- Show source chunks and page references
- Keep simple session chat history in SQLite
- Clear chat history from the UI or API
- Handle unsupported files, empty uploads, parser failures, missing API keys, and LLM errors

## Architecture

```text
frontend/streamlit_app.py
        |
        v
backend/app/main.py
        |
        +-- routes/upload.py -> parser -> chunker -> embeddings -> Chroma
        |
        +-- routes/chat.py   -> retriever -> RAG prompt -> Gemini -> SQLite history
```

The project is intentionally modular without being over-engineered. Routes handle HTTP concerns, services handle RAG logic, models define API shapes, and SQLite is kept in a small database helper module.

Note: Chroma is the intended vector database. On Windows with Python 3.12, some Chroma native wheels can crash during local inserts. The app automatically uses a small SQLite vector fallback in that environment so the MVP still runs. To force Chroma, set `VECTOR_STORE_BACKEND=chroma` and use Python 3.11 or install the required native build tools.

## Project Structure

```text
rag-study-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── upload.py
│   │   │   └── chat.py
│   │   ├── services/
│   │   │   ├── parser.py
│   │   │   ├── chunker.py
│   │   │   ├── embeddings.py
│   │   │   ├── vector_store.py
│   │   │   ├── rag_pipeline.py
│   │   │   └── llm.py
│   │   ├── db/
│   │   │   └── database.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   └── utils/
│   │       └── helpers.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   └── streamlit_app.py
├── uploads/
├── chroma_db/
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Create `backend/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-pro
```

Optional tuning for free-tier rate limits:

```env
EMBEDDING_BATCH_SIZE=20
EMBEDDING_BATCH_PAUSE_SECONDS=2
MAX_EMBEDDING_RETRIES=5
MAX_LLM_RETRIES=3
```

## Run the Backend

From the project root:

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive docs:

```text
http://localhost:8000/docs
```

## Run the Frontend

In another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

If your backend runs somewhere else, set:

```bash
set API_BASE_URL=http://localhost:8000
```

## API Endpoints

### `POST /upload`

Uploads and indexes one file. Each upload replaces the currently active study material in this MVP.

### `POST /chat`

Request:

```json
{
  "question": "What is photosynthesis?"
}
```

Response:

```json
{
  "answer": "Photosynthesis is...",
  "sources": [
    {
      "content": "source chunk text",
      "source": "biology_notes.pdf",
      "page": 3,
      "chunk_id": "chunk-5"
    }
  ]
}
```

### `POST /clear-chat`

Clears SQLite chat history for the local session.

## Screenshots

Add screenshots here after running the app:

- `docs/screenshots/upload.png`
- `docs/screenshots/chat.png`
- `docs/screenshots/sources.png`

## Deployment Notes

For Render, deploy the FastAPI backend as a web service with:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Add `GEMINI_API_KEY` in Render environment variables. Streamlit can be deployed separately, with `API_BASE_URL` pointing to the backend URL.

## Future Improvements

- Multiple document collections instead of replacing the active upload
- User accounts and persistent per-user chat history
- Better PDF table extraction
- Streaming answers in the UI
- Dockerfile and docker-compose setup
- Downloadable citations or study summaries
- Reranking for higher retrieval quality
