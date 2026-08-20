"""
Session-based RAG engine.

Unlike the original CollegeRAG (which loaded one fixed dataset at startup),
this version lets each "session" be built from whatever files a user
uploads at request time. Every session gets:

  - its own Qdrant collection (isolated vector storage)
  - its own BM25 index (built from that session's chunks)
  - its own hybrid retriever (BM25 + Qdrant + BGE rerank)

so answers are always grounded only in the documents that particular user
uploaded for that particular session - never mixed with anyone else's data
or any previous file.
"""

from typing import Dict, List, Optional

from .chunker import create_chunks
from .embeddings import load_embeddings
from .llm import load_llm
from .loader import load_documents_from_paths
from .prompt import prompt
from .retriever import get_retriever
from .vectorstore import create_vectorstore, new_collection_name, delete_collection


class RAGSession:
    """Everything needed to answer questions about one set of uploaded files."""

    def __init__(self, session_id: str, collection_name: str, retriever, sources: List[str]):
        self.session_id = session_id
        self.collection_name = collection_name
        self.retriever = retriever
        self.sources = sources  # original file names, for reference/citation


class CollegeRAG:
    """
    Manages the shared embedding model + LLM, and creates/holds per-session
    retrieval pipelines keyed by session_id. This class is created once
    (e.g. as a FastAPI app-level singleton) and reused across requests.
    """

    def __init__(self):
        # Embedding model and LLM are expensive to load, so they're shared
        # across all sessions.
        self.embedding = load_embeddings()
        self.llm = load_llm()
        self.sessions: Dict[str, RAGSession] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_files(self, session_id: str, file_paths: List[str]) -> RAGSession:
        """
        Load, chunk, and index the given files (PDF and/or TXT) under a
        fresh Qdrant collection for this session. If the session already
        exists, its old collection is replaced with the new files.
        """
        if not file_paths:
            raise ValueError("No files provided to ingest.")

        # Clean up any previous collection for this session_id so re-uploads
        # don't leak old context into new answers.
        existing = self.sessions.get(session_id)
        if existing:
            delete_collection(existing.collection_name)

        documents = load_documents_from_paths(file_paths)
        if not documents:
            raise ValueError(
                "No readable text was extracted from the uploaded file(s)."
            )

        chunks = create_chunks(documents)

        collection_name = new_collection_name(prefix=session_id)
        vectorstore = create_vectorstore(chunks, self.embedding, collection_name)

        retriever = get_retriever(chunks, vectorstore, final_k=5, use_reranker=True)

        source_names = sorted({doc.metadata.get("source", "unknown") for doc in documents})

        session = RAGSession(
            session_id=session_id,
            collection_name=collection_name,
            retriever=retriever,
            sources=source_names,
        )
        self.sessions[session_id] = session
        return session

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def ask(self, session_id: str, question: str):
        """
        Answer a question using only the documents ingested for this
        session_id. Raises ValueError if no files have been uploaded yet
        for this session.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(
                "No documents have been uploaded for this session yet. "
                "Please upload a PDF or TXT file first."
            )

        docs = session.retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in docs)

        final_prompt = prompt.invoke({
            "context": context,
            "question": question,
        })

        answer = self.llm.invoke(final_prompt)

        return {
            "answer": answer,
            "sources": session.sources,
            "retrieved_chunks": [
                {
                    "source": doc.metadata.get("source"),
                    "page_number": doc.metadata.get("page_number"),
                    "content": doc.page_content,
                }
                for doc in docs
            ],
        }

    def has_session(self, session_id: str) -> bool:
        return session_id in self.sessions

    def end_session(self, session_id: str) -> None:
        """Remove a session and its Qdrant collection."""
        session = self.sessions.pop(session_id, None)
        if session:
            delete_collection(session.collection_name)
