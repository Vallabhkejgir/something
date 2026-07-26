import asyncio
import logging
from typing import List, Optional
import os
import pickle

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from app.services.llm_config import embeddings

logger = logging.getLogger(__name__)

# Persistent Qdrant local client
qdrant_client = None

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(path="app/services/qdrant_data")
    return qdrant_client
COLLECTION_NAME = "employee_knowledge_base"


class VectorStoreManager:
    """
    Manages Qdrant, BM25, and parent document stores.

    The `_lock` ensures that concurrent indexing requests (multiple tabs)
    do not corrupt internal state. Single-writer, multi-reader pattern.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        # Vector stores
        self.vector_store: Optional[QdrantVectorStore] = None
        self.bm25_retriever: Optional[BM25Retriever] = None

        # Parent doc store: chunk_id -> Document (full parent text)
        self.parent_store: dict[str, Document] = {}

        # Accumulator of ALL indexable (child) docs — needed to rebuild BM25
        self._all_index_docs: List[Document] = []

        # Tracking
        self.indexed_urls: set[str] = set()
        self._index_version: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def add_documents(self, docs: List[Document], url: Optional[str] = None) -> None:
        """
        Ingest a batch of documents (parents + children).

        Parents (is_parent=True) are routed to the parent_store only.
        Children / standalone docs are indexed in Qdrant and BM25.
        """
        if not docs:
            logger.warning("add_documents called with empty list — skipping.")
            if url:
                self.indexed_urls.add(url)
            return

        async with self._lock:
            parent_docs = [d for d in docs if d.metadata.get("is_parent")]
            index_docs  = [d for d in docs if not d.metadata.get("is_parent")]

            # Route parents to parent store
            for pd in parent_docs:
                cid = pd.metadata.get("chunk_id")
                if cid:
                    self.parent_store[cid] = pd

            if not index_docs:
                logger.info("No indexable (child) docs — parents stored only.")
                if url:
                    self.indexed_urls.add(url)
                return

            # Accumulate all index docs (needed for BM25 full rebuild)
            self._all_index_docs.extend(index_docs)

            # Update Qdrant
            logger.info("Indexing %d docs into Qdrant...", len(index_docs))
            if self.vector_store is None:
                self.vector_store = QdrantVectorStore(
                    client=get_qdrant_client(),
                    collection_name=COLLECTION_NAME,
                    embedding=embeddings,
                )

            await asyncio.to_thread(self.vector_store.add_documents, index_docs)

            # Rebuild BM25 from ALL accumulated docs
            logger.info("Rebuilding BM25 over %d total docs...", len(self._all_index_docs))
            bm25 = await asyncio.to_thread(
                BM25Retriever.from_documents, self._all_index_docs
            )
            bm25.k = 10
            self.bm25_retriever = bm25

            self._index_version += 1
            if url:
                self.indexed_urls.add(url)

            logger.info(
                "Indexing complete — v%d | %d parents | %d indexed | %d URLs",
                self._index_version,
                len(self.parent_store),
                len(self._all_index_docs),
                len(self.indexed_urls),
            )

    def get_parent(self, child_doc: Document) -> Document:
        """
        Given a child doc, return its parent from the parent_store.
        Falls back to the child itself if no parent is found.
        """
        parent_id = child_doc.metadata.get("parent_chunk_id")
        if parent_id and parent_id in self.parent_store:
            return self.parent_store[parent_id]
        return child_doc

    def is_initialized(self) -> bool:
        return self.vector_store is not None

    def is_url_indexed(self, url: str) -> bool:
        return url in self.indexed_urls

    @property
    def index_version(self) -> int:
        return self._index_version

    def stats(self) -> dict:
        return {
            "initialized": self.is_initialized(),
            "indexed_urls": list(self.indexed_urls),
            "total_indexed_docs": len(self._all_index_docs),
            "total_parents": len(self.parent_store),
            "index_version": self._index_version,
        }

    def get_vector_store(self):
        if self.vector_store is None:
            try:
                self.vector_store = QdrantVectorStore(
                    client=get_qdrant_client(),
                    collection_name=COLLECTION_NAME,
                    embedding=embeddings,
                )
            except Exception:
                pass
        return self.vector_store

    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        """
        Combines dense and sparse search results using Reciprocal Rank Fusion.
        """
        rrf_scores = {}

        for rank, doc in enumerate(dense_results):
            doc_key = doc.metadata.get("chunk_id", doc.page_content)
            if doc_key not in rrf_scores:
                rrf_scores[doc_key] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_key]["score"] += 1.0 / (rank + 1 + k)

        for rank, doc in enumerate(sparse_results):
            doc_key = doc.metadata.get("chunk_id", doc.page_content)
            if doc_key not in rrf_scores:
                rrf_scores[doc_key] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_key]["score"] += 1.0 / (rank + 1 + k)

        sorted_results = sorted(list(rrf_scores.values()), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results]


# ── Singleton ─────────────────────────────────────────────────────────────────
store_manager = VectorStoreManager()

# ── Legacy shim for backwards-compat (used in old nodes.py imports) ───────────
def store_chunks(splits, url=None):
    """Backwards-compatible sync wrapper. Use store_manager.add_documents() directly."""
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    try:
        loop.run_until_complete(store_manager.add_documents(splits, url=url))
    finally:
        loop.close()

indexed_urls = store_manager.indexed_urls
