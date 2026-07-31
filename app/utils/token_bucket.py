import asyncio
import time

class TokenBucket:
    def __init__(self, max_tokens_per_min, max_requests_per_min):
        self.max_tokens = float(max_tokens_per_min)
        self.tokens = float(max_tokens_per_min)
        self.max_requests = float(max_requests_per_min)
        self.requests = float(max_requests_per_min)
        self.updated_at = time.time()
        self._lock = None

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self, tokens_needed=1):
        if tokens_needed > self.max_tokens:
            raise ValueError(f"tokens_needed ({tokens_needed}) exceeds max_tokens ({self.max_tokens})")
        while True:
            wait_time = 0.0
            async with self.lock:
                now = time.time()
                elapsed = now - self.updated_at
                
                refill_rate_tokens = self.max_tokens / 60.0
                refill_rate_requests = self.max_requests / 60.0
                
                self.tokens = min(self.max_tokens, self.tokens + (elapsed * refill_rate_tokens))
                self.requests = min(self.max_requests, self.requests + (elapsed * refill_rate_requests))
                self.updated_at = now

                if self.tokens < tokens_needed or self.requests < 1:
                    deficit_tokens = max(0, tokens_needed - self.tokens)
                    token_wait = deficit_tokens / refill_rate_tokens if refill_rate_tokens > 0 else 0.0
                    
                    deficit_reqs = max(0, 1 - self.requests)
                    req_wait = deficit_reqs / refill_rate_requests if refill_rate_requests > 0 else 0.0
                    
                    wait_time = max(token_wait, req_wait)

                if wait_time <= 0:
                    self.tokens -= tokens_needed
                    self.requests -= 1
                    return

            # Sleep outside the lock so other coroutines are not blocked
            await asyncio.sleep(wait_time)
