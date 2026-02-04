import asyncio
import time

class TokenBucket:
    def __init__(self, max_tokens_per_min, max_requests_per_min):
        self.max_tokens = max_tokens_per_min
        self.tokens = max_tokens_per_min
        self.updated_at = time.time()
        self.request_interval = 60.0 / max_requests_per_min
        self.last_request_time = 0

    async def acquire(self, tokens_needed):
        while True:
            now = time.time()
            elapsed = now - self.updated_at
            refill_rate = self.max_tokens / 60.0
            self.tokens = min(self.max_tokens, self.tokens + (elapsed * refill_rate))
            self.updated_at = now

            time_since_last_req = now - self.last_request_time
            if time_since_last_req < self.request_interval:
                await asyncio.sleep(self.request_interval - time_since_last_req)
                continue

            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                self.last_request_time = time.time()
                return
            else:
                deficit = tokens_needed - self.tokens
                wait_time = deficit / refill_rate
                await asyncio.sleep(max(wait_time, 1.0))