"""
Hybrid retrieval: BM25 (keyword) + Qdrant (semantic/dense) search, fused and
then re-ranked with a BGE cross-encoder reranker for higher-quality context.

Pipeline for a query:
  1. BM25Retriever over the in-memory chunks -> keyword-relevant docs
  2. Qdrant similarity search -> semantically-relevant docs
  3. Union/de-duplicate both result sets ("hybrid retrieval")
  4. Re-rank the fused set with a BGE cross-encoder reranker
  5. Return the top_k most relevant chunks as final context
"""

from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

_reranker = None


def _get_reranker():
    """
    Lazily load the BGE reranker (cross-encoder) model. Loaded once and
    cached, since loading a HF model is relatively expensive.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


class HybridRetriever:
    """
    Combines BM25 keyword retrieval with Qdrant dense semantic retrieval,
    then reranks the fused candidate set with a BGE cross-encoder.

    One HybridRetriever instance is created per uploaded document set
    (per session), since BM25 needs the raw chunk texts in memory.
    """

    def __init__(
        self,
        chunks: List[Document],
        vectorstore,
        bm25_k: int = 10,
        semantic_k: int = 10,
        final_k: int = 5,
        use_reranker: bool = True,
    ):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.bm25_k = bm25_k
        self.semantic_k = semantic_k
        self.final_k = final_k
        self.use_reranker = use_reranker

        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = bm25_k

    def _fuse(self, query: str) -> List[Document]:
        bm25_docs = self.bm25_retriever.invoke(query)
        semantic_docs = self.vectorstore.similarity_search(query, k=self.semantic_k)

        seen = set()
        fused: List[Document] = []

        for doc in bm25_docs + semantic_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                fused.append(doc)

        return fused

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs

        if not self.use_reranker:
            return docs[: self.final_k]

        try:
            reranker = _get_reranker()
        except Exception:
            # If the reranker model can't be loaded (e.g. no internet on
            # first download), gracefully fall back to the fused order.
            return docs[: self.final_k]

        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)

        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[: self.final_k]]

    def invoke(self, query: str) -> List[Document]:
        fused = self._fuse(query)
        return self._rerank(query, fused)

    # Kept for compatibility with code that expects a `.get_relevant_documents`
    # style call (older LangChain retriever interface).
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)


def get_retriever(
    chunks: List[Document],
    vectorstore,
    final_k: int = 5,
    use_reranker: bool = True,
) -> HybridRetriever:
    """Build a HybridRetriever (BM25 + Qdrant semantic + BGE rerank) for a session."""
    return HybridRetriever(
        chunks=chunks,
        vectorstore=vectorstore,
        final_k=final_k,
        use_reranker=use_reranker,
    )
