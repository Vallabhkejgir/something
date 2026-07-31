import re

with open("app/services/cache.py", "r") as f:
    content = f.read()

new_get = """    async def get(self, query: str, index_version: int) -> Optional[dict]:
        \"\"\"
        Look up a cached result for the query.
        Returns the cached result dict or None on miss.
        \"\"\"
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
            query_vec = await asyncio.to_thread(_qe.embed_query, query)
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
        return None"""

new_set = """    async def set(self, query: str, result: dict, index_version: int) -> None:
        \"\"\"Store a query result in the cache.\"\"\"
        try:
            from app.services.llm_config import embeddings as _qe
            query_vec = await asyncio.to_thread(_qe.embed_query, query)
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
                self._store = self._store[-_MAX_CACHE_SIZE:]"""

content = re.sub(r'    async def get\(self, query: str, index_version: int\) -> Optional\[dict\]:.*?        return None', new_get, content, flags=re.DOTALL)
content = re.sub(r'    async def set\(self, query: str, result: dict, index_version: int\) -> None:.*?(?=    def invalidate\(self\))', new_set + "\n\n", content, flags=re.DOTALL)

with open("app/services/cache.py", "w") as f:
    f.write(content)

