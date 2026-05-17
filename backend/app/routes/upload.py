"""File upload endpoint."""

import traceback
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from backend.app.db.database import clear_chat_history
from backend.app.models.schemas import UploadResponse
from backend.app.services.chunker import chunk_documents
from backend.app.services.parser import DocumentParsingError, parse_document
from backend.app.services.vector_store import reset_vector_store, store_chunks
from backend.app.utils.helpers import UPLOAD_DIR, ensure_project_dirs, is_supported_file, safe_upload_filename


router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Save, parse, chunk, and embed one uploaded study document."""

    print("\n===== UPLOAD PIPELINE START =====")
    ensure_project_dirs()

    if not file.filename or not is_supported_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Please upload a PDF, TXT, or DOCX file.",
        )

    contents = await file.read()
    print(f"Received file: {file.filename}")
    print(f"Uploaded file size: {len(contents)} bytes")
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    saved_path = UPLOAD_DIR / safe_upload_filename(file.filename)
    try:
        saved_path.write_bytes(contents)
        print(f"File uploaded successfully: {saved_path}")
    except Exception as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the uploaded file.",
        ) from exc

    try:
        print("Parsing document...")
        documents = parse_document(Path(saved_path))
        total_text_length = sum(len(document.page_content) for document in documents)
        print("Parsing complete")
        print(f"Parsed {len(documents)} document sections")
        print(f"Extracted text length: {total_text_length}")

        print("Chunking document...")
        chunks = chunk_documents(documents)
        print(f"Generated {len(chunks)} chunks")
        if not chunks:
            raise DocumentParsingError("No text chunks could be created from this file.")


        print("STEP 4: Starting vector storage")

        try:
            print("STEP 4.1: Calling store_chunks()")

            store_chunks(chunks)

            print("STEP 4.2: store_chunks() completed successfully")

        except Exception as vector_error:
            print("\n===== VECTOR STORE ERROR =====")
            print(str(vector_error))
            traceback.print_exc()

            raise RuntimeError(
                f"Vector store failed: {str(vector_error)}"
            ) from vector_error

        print("STEP 5: Upload indexing completed")

    except DocumentParsingError as exc:
        print("\n===== ERROR =====")
        print(str(exc))
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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

    print("===== UPLOAD PIPELINE END =====\n")
    return UploadResponse(
        message="File uploaded and indexed successfully.",
        filename=saved_path.name,
        chunks_created=len(chunks),
    )
