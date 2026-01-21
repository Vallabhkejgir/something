import asyncio
import time

class TokenBucket:
    """
    Manages the API quotas (TPM and RPM) safely for Async tasks.
    """
    def __init__(self, max_tokens_per_min, max_requests_per_min):
        self.max_tokens = max_tokens_per_min
        self.tokens = max_tokens_per_min
        self.updated_at = time.time()
        
        # RPM (Requests Per Minute) handling
        self.request_interval = 60.0 / max_requests_per_min
        self.last_request_time = 0

    async def acquire(self, tokens_needed):
        """
        Waits until enough tokens are available.
        """
        while True:
            now = time.time()
            
            # 1. Refill the bucket based on time passed
            elapsed = now - self.updated_at
            refill_rate = self.max_tokens / 60.0  # Tokens per second
            self.tokens = min(self.max_tokens, self.tokens + (elapsed * refill_rate))
            self.updated_at = now

            # 2. Check RPM (Requests Per Minute) safety
            time_since_last_req = now - self.last_request_time
            if time_since_last_req < self.request_interval:
                await asyncio.sleep(self.request_interval - time_since_last_req)
                continue

            # 3. Check TPM (Tokens Per Minute) availability
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                self.last_request_time = time.time()
                return  # Success! Proceed to API call.
            else:
                # Not enough tokens? Wait a bit and try again.
                # Calc how long to wait for refill:
                deficit = tokens_needed - self.tokens
                wait_time = deficit / refill_rate
                # Wait at least 1 second to avoid busy-looping
                await asyncio.sleep(max(wait_time, 1.0))
