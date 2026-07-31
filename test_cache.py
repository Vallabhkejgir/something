import asyncio
from app.api import query, QueryRequest
from app.services.storage import store_manager
from app.services.cache import query_cache

async def test_cache():
    # Mock initialized
    import app.api as api
    api.initialized = True
    
    class MockDoc:
        page_content = "The sky is blue."
        metadata = {"chunk_id": "1"}
        
    store_manager._index_version = 1
    
    # Pre-populate cache
    # Note: Because embedding fails without an API key, we will simulate set by appending directly.
    query_cache._store.append({
        "original_query": "What color is the sky?",
        "embedding": [0.1]*3072,
        "result": {"answer": "Blue."},
        "index_version": 1,
        "timestamp": 999999999999, # Far in future so it doesn't expire
    })
    
    # Query with exactly the same string (should hit exact match fast path)
    req = QueryRequest(prompt="what color is the sky?")
    res = await query(req)
    print("Exact Match:", res)

if __name__ == "__main__":
    asyncio.run(test_cache())
