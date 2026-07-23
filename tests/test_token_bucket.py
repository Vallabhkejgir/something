import asyncio
import time
import pytest
from app.utils.token_bucket import TokenBucket

@pytest.mark.anyio
async def test_token_bucket_basic():
    bucket = TokenBucket(max_tokens_per_min=60, max_requests_per_min=600)
    start = time.time()
    await bucket.acquire(10)
    elapsed = time.time() - start
    assert elapsed < 0.1
    assert bucket.tokens <= 50.0

@pytest.mark.anyio
async def test_token_bucket_concurrent_requests():
    bucket = TokenBucket(max_tokens_per_min=600, max_requests_per_min=600)
    acquired_count = 0

    async def worker():
        nonlocal acquired_count
        await bucket.acquire(1)
        acquired_count += 1

    tasks = [asyncio.create_task(worker()) for _ in range(10)]
    await asyncio.gather(*tasks)
    assert acquired_count == 10

@pytest.mark.anyio
async def test_token_bucket_lock_release_during_sleep():
    bucket = TokenBucket(max_tokens_per_min=60, max_requests_per_min=2)
    await bucket.acquire(1)
    assert not bucket.lock.locked()
