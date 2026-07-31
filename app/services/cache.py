"""
cache.py — Semantic query result cache.

Caches query results using the query's embedding as the cache key.
This catches semantically similar queries (not just exact matches).

Implementation:
  - Embeddings are computed with `query_embeddings` (task_type=retrieval_query).
  - Cosine similarity is used to find near-duplicate cached queries.
  - Cache is invalidated whenever the index version changes.
  - TTL ensures stale entries are evicted over time.

Usage:
    cache = QueryCache()
    hit = await cache.get(query, store_manager.index_version)
    if hit: return hit
    # ... compute result ...
    await cache.set(query, result, store_manager.index_version)
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_TTL         = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
_SIMILARITY_THRESH = 0.97   # Cosine similarity threshold for a cache hit
_MAX_CACHE_SIZE    = 200    # Maximum number of cached entries


class QueryCache:
    """
    Semantic cache keyed by (query_embedding, index_version).

    Structure per entry:
      {
        "embedding": List[float],
        "result":    dict,
        "index_version": int,
        "timestamp": float,
      }
    """

    def __init__(self):
        self._store: list[dict] = []
        self._lock = asyncio.Lock()

    async def get(self, query: str, index_version: int) -> Optional[dict]:
        """
        Look up a cached result for the query.
        Returns the cached result dict or None on miss.
        """
        if not self._store:
            return None

        now = time.time()
        
        # 1. Fast path: exact string match (case-insensitive)
        async with self._lock:
            for entry in self._store:
                if entry["index_version"] != index_version:
                    continue
                if now - entry["timestamp"] > _CACHE_TTL:
                    continue
                if entry.get("original_query", "").strip().lower() == query.strip().lower():
                    logger.info("Cache HIT (exact match) for query: %.60s", query)
                    return entry["result"]

        # 2. Slow path: semantic search
        try:
            from app.services.llm_config import embeddings as _qe
            query_vec = await _qe.aembed_query(query)
        except Exception as e:
            logger.warning("Cache: embedding failed for lookup (%s)", e)
            return None

        async with self._lock:
            for entry in self._store:
                if entry["index_version"] != index_version:
                    continue
                if now - entry["timestamp"] > _CACHE_TTL:
                    continue
                sim = _cosine_similarity(query_vec, entry["embedding"])
                if sim >= _SIMILARITY_THRESH:
                    logger.info("Cache HIT (similarity=%.4f) for query: %.60s", sim, query)
                    return entry["result"]

        logger.debug("Cache MISS for query: %.60s", query)
        return None

        try:
            from app.services.llm_config import embeddings as _qe
            query_vec = await _qe.aembed_query(query)
        except Exception as e:
            logger.warning("Cache: embedding failed for lookup (%s)", e)
            return None

        now = time.time()
        async with self._lock:
            for entry in self._store:
                # Version guard: cached result from an older index is stale
                if entry["index_version"] != index_version:
                    continue
                # TTL guard
                if now - entry["timestamp"] > _CACHE_TTL:
                    continue
                # Similarity check
                sim = _cosine_similarity(query_vec, entry["embedding"])
                if sim >= _SIMILARITY_THRESH:
                    logger.info("Cache HIT (similarity=%.4f) for query: %.60s", sim, query)
                    return entry["result"]

        logger.debug("Cache MISS for query: %.60s", query)
        return None

    async def set(self, query: str, result: dict, index_version: int) -> None:
        """Store a query result in the cache."""
        try:
            from app.services.llm_config import embeddings as _qe
            query_vec = await _qe.aembed_query(query)
        except Exception as e:
            logger.warning("Cache: embedding failed for set (%s) — not caching.", e)
            return

        async with self._lock:
            self._store.append({
                "original_query": query,
                "embedding":     query_vec,
                "result":        result,
                "index_version": index_version,
                "timestamp":     time.time(),
            })
            # Evict oldest entries if over size limit
            if len(self._store) > _MAX_CACHE_SIZE:
                self._store = self._store[-_MAX_CACHE_SIZE:]

    def invalidate(self) -> None:
        """Clear the entire cache (call after re-indexing)."""
        self._store.clear()
        logger.info("Cache invalidated.")

    def stats(self) -> dict:
        return {
            "entries": len(self._store),
            "ttl_seconds": _CACHE_TTL,
            "max_size": _MAX_CACHE_SIZE,
        }


# ── Cosine Similarity ─────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Singleton ─────────────────────────────────────────────────────────────────
query_cache = QueryCache()
