"""
FastAPI backend for the document-agnostic RAG chatbot.

Endpoints
---------
POST /session                 -> create a new session_id
POST /upload/{session_id}     -> upload one or more PDF/TXT files, indexes them
POST /ask/{session_id}        -> ask a question, answered ONLY from that
                                  session's uploaded documents
GET  /session/{session_id}    -> check whether a session has documents ready
DELETE /session/{session_id}  -> delete a session and its vector data
GET  /health                  -> basic health check

Run with:
    uvicorn api:app --reload --port 8000
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.loader import SUPPORTED_EXTENSIONS
from src.rag import CollegeRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("college_bot_api")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Document RAG Chatbot API",
    description=(
        "Upload any PDF or TXT file and ask questions about it. "
        "Each session is isolated, so different uploads never mix context."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global safety net: make sure the client ALWAYS gets valid JSON back, even
# for unexpected/unhandled errors. Without this, FastAPI/Starlette's default
# 500 handler returns a *plain text* body ("Internal Server Error"), which
# breaks any client that blindly calls resp.json() (that's what was causing
# `requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1`
# in app.py - the API had crashed and sent back non-JSON text).
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


# Single shared RAG engine (embedding model + LLM loaded once), with
# per-session retrieval pipelines managed internally.
try:
    rag_engine = CollegeRAG()
except Exception:
    logger.exception(
        "Failed to initialize CollegeRAG (embedding/LLM model loading). "
        "Check your .env API keys and internet connection for first-time "
        "model downloads."
    )
    raise


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SessionResponse(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    files_ingested: List[str]
    chunk_count: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


class SessionStatus(BaseModel):
    session_id: str
    ready: bool
    sources: List[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_answer_text(answer) -> str:
    """
    Different LLM backends shape `.content` differently:
      - Gemini/Groq (Chat* wrappers): content is a plain string
      - Some providers: content is a list of dicts like [{"type": "text", "text": "..."}]
      - Our FallbackLLM: content is [{"text": "..."}]
    """
    content = getattr(answer, "content", answer)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)

    return str(content)


def _validate_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session", response_model=SessionResponse)
def create_session():
    """Create a new, empty session. Upload files to it, then ask questions."""
    session_id = uuid.uuid4().hex[:12]
    (UPLOAD_DIR / session_id).mkdir(parents=True, exist_ok=True)
    return SessionResponse(session_id=session_id)


@app.post("/upload/{session_id}", response_model=UploadResponse)
async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF/TXT files for this session. This (re)builds the
    session's vector index from scratch using ONLY the files provided here,
    so the bot only ever answers from what was just uploaded.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for upload in files:
        _validate_extension(upload.filename)
        dest_path = session_dir / upload.filename
        with dest_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(str(dest_path))

    try:
        session = rag_engine.ingest_files(session_id, saved_paths)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Failed to process file(s): {e}")

    return UploadResponse(
        session_id=session_id,
        files_ingested=session.sources,
        chunk_count=len(session.retriever.chunks),
    )


@app.post("/ask/{session_id}", response_model=AskResponse)
def ask_question(session_id: str, request: AskRequest):
    """Ask a question, answered strictly from this session's uploaded documents."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = rag_engine.ask(session_id, request.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Catches LLM provider errors (bad/expired API key, rate limits,
        # network issues, decommissioned model names, etc.), retriever
        # errors, and anything else so the client always gets valid JSON
        # back instead of a crashed connection / plain-text 500.
        logger.exception("Failed to answer question for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Failed to generate an answer: {e}")

    try:
        answer_text = _extract_answer_text(result["answer"])
    except Exception as e:
        logger.exception("Failed to parse LLM response for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Failed to parse the model's response: {e}")

    return AskResponse(answer=answer_text, sources=result["sources"])


@app.get("/session/{session_id}", response_model=SessionStatus)
def session_status(session_id: str):
    ready = rag_engine.has_session(session_id)
    sources = rag_engine.sessions[session_id].sources if ready else []
    return SessionStatus(session_id=session_id, ready=ready, sources=sources)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    rag_engine.end_session(session_id)
    session_dir = UPLOAD_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    return {"status": "deleted", "session_id": session_id}
