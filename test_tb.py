import asyncio
import time
from app.utils.token_bucket import TokenBucket

async def main():
    tb = TokenBucket(max_tokens_per_min=60, max_requests_per_min=60)
    
    async def worker(id):
        print(f"Worker {id} waiting...")
        await tb.acquire(1)
        print(f"Worker {id} acquired at {time.time()}")

    start = time.time()
    await asyncio.gather(*(worker(i) for i in range(5)))
    print(f"Total time: {time.time() - start}")

asyncio.run(main())
