# College Bot — Multi-Document RAG Chatbot

Upload **any** PDF or TXT file(s) and ask questions about them. Each upload
replaces the previous one, so the bot always answers strictly from whatever
document(s) you most recently gave it — it is no longer hard-coded to a
single dataset.

## What changed from the original version

| Feature | Before | Now |
|---|---|---|
| Data source | One fixed `data/data_tnu.txt` file, baked in at index-build time | Any PDF/TXT file(s) uploaded by the user, at any time |
| Vector DB | Chroma | **Qdrant** (embedded, on-disk — no server required) |
| Retrieval | Pure semantic (dense vector) search | **Hybrid**: BM25 keyword search + Qdrant semantic search, fused |
| Ranking | Raw vector similarity order | **BGE cross-encoder reranker** re-scores the fused candidates |
| Backend | None (Streamlit called the RAG class directly) | **FastAPI** backend (`api.py`) with upload/ask/session endpoints |
| Frontend | Streamlit called the bot in-process | **Streamlit** (`app.py`) is now a client that calls the FastAPI backend over HTTP |
| Multi-user | Single global index | Each upload gets its own isolated session + Qdrant collection — different users'/uploads' documents never mix |

## Project structure

```
College_Bot/
├── api.py                 # FastAPI backend (upload, ask, session endpoints)
├── app.py                 # Streamlit UI (talks to api.py over HTTP)
├── requirements.txt
├── .env                   # API keys (GROQ_API_KEY, etc.)
├── src/
│   ├── loader.py           # Loads PDF/TXT files given at runtime
│   ├── chunker.py          # Splits documents into chunks
│   ├── embeddings.py       # HuggingFace embedding model
│   ├── vectorstore.py      # Qdrant collection management (per-session)
│   ├── retriever.py        # Hybrid BM25 + Qdrant + BGE reranker
│   ├── llm.py               # Groq LLM loader (with fallback)
│   ├── prompt.py            # Prompt template + safety rules
│   └── rag.py                # CollegeRAG: session manager tying it all together
├── uploads/                # Uploaded files land here, per session_id
└── qdrant_storage/          # Local on-disk Qdrant data
```

## Setup

```bash
pip install -r requirements.txt
```

Make sure your `.env` has at least one working LLM key (`GROQ_API_KEY` is
used by default). Without it, the bot will still run but respond with a
"chatbot unavailable" fallback message.

The first time you ask a question, the BGE reranker
(`BAAI/bge-reranker-base`) and the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`) will be downloaded from
HuggingFace automatically (requires internet access on first run).

## Running it

**1. Start the FastAPI backend:**

```bash
uvicorn api:app --reload --port 8000
```

You can explore/test the API directly at `http://localhost:8000/docs`
(interactive Swagger UI).

**2. In a separate terminal, start the Streamlit UI:**

```bash
streamlit run app.py
```

Then in the browser:
1. Upload one or more PDF/TXT files in the sidebar.
2. Click **Process file(s)** — this chunks, embeds, and indexes them into a
   fresh Qdrant collection for hybrid + reranked retrieval.
3. Ask questions in the chat box. Answers are generated **only** from the
   uploaded file(s).
4. To ask about a different document, just upload a new file — it replaces
   the old one for that session automatically.

## API reference (if you want to call it directly, e.g. from another app)

| Method | Path | Description |
|---|---|---|
| POST | `/session` | Create a new empty session, returns `session_id` |
| POST | `/upload/{session_id}` | Upload one or more PDF/TXT files (multipart `files`) |
| POST | `/ask/{session_id}` | `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |
| GET | `/session/{session_id}` | Check if a session has documents indexed |
| DELETE | `/session/{session_id}` | Remove a session and its Qdrant data |
| GET | `/health` | Health check |

## Notes

- Qdrant runs in **embedded/local mode** by default (data stored in
  `qdrant_storage/`) — no separate Qdrant server needs to be installed or
  running. If you'd rather point at a real Qdrant server (Docker or Qdrant
  Cloud), set `QDRANT_URL` (and optionally `QDRANT_API_KEY`) in `.env`.
- Re-uploading files to the same `session_id` deletes that session's old
  Qdrant collection and rebuilds it from the new file(s), so stale context
  never lingers.
- If the BGE reranker model can't be downloaded (e.g. no internet), the
  hybrid retriever gracefully falls back to un-reranked fused results
  instead of crashing.
