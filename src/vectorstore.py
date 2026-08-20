"""
Qdrant vector store integration.

Each "session" (i.e. each set of files a user uploads) gets its own Qdrant
collection, so different users / different uploads never mix context. We
run Qdrant in local, embedded (on-disk) mode via `path=...`, so no separate
Qdrant server needs to be running. If a QDRANT_URL is configured (e.g. for
a hosted/self-hosted Qdrant server), that is used instead.
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

BASE_DIR = Path(__file__).resolve().parents[1]
LOCAL_QDRANT_PATH = str(BASE_DIR / "qdrant_storage")

QDRANT_URL = os.getenv("QDRANT_URL")  # e.g. http://localhost:6333, optional
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # optional, for Qdrant Cloud

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """
    Return a singleton QdrantClient.

    - If QDRANT_URL is set, connect to that Qdrant server (local Docker
      instance or Qdrant Cloud).
    - Otherwise fall back to an embedded, on-disk Qdrant instance, which
      needs no separate server/install and works out of the box.
    """
    global _client

    if _client is not None:
        return _client

    if QDRANT_URL:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        _client = QdrantClient(path=LOCAL_QDRANT_PATH)

    return _client


def new_collection_name(prefix: str = "session") -> str:
    """Generate a fresh, unique collection name for a new upload/session."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def create_vectorstore(
    chunks: List[Document],
    embedding,
    collection_name: str,
) -> QdrantVectorStore:
    """
    Create (or recreate) a Qdrant collection from the given chunks and
    return a LangChain QdrantVectorStore wrapping it.
    """
    client = get_client()

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    vector_size = len(embedding.embed_query("dimension probe"))

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )

    store.add_documents(chunks)

    return store


def load_vectorstore(embedding, collection_name: str) -> QdrantVectorStore:
    """Load an existing Qdrant collection as a LangChain QdrantVectorStore."""
    client = get_client()

    if not client.collection_exists(collection_name):
        raise ValueError(
            f"Collection '{collection_name}' does not exist. Upload a file first."
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )


def delete_collection(collection_name: str) -> None:
    """Remove a collection (e.g. to clean up after a session ends)."""
    client = get_client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)


def collection_exists(collection_name: str) -> bool:
    client = get_client()
    return client.collection_exists(collection_name)
