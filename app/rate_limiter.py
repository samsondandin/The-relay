"""Sliding-window Rate Limiter and Daily Budget Manager.

Policy Decision:
We implement a Sliding-Window Log policy using atomic double-ended queues (deques)
guarded by an asyncio.Lock. Sliding-window prevents the 'boundary burst' vulnerability
inherent in fixed-window limiters (where double the quota can arrive across the window edge).
Note: This local limiter regulates client traffic; it does not claim to reproduce Groq's
exact internal proprietary distributed token-bucket or lease-based algorithms.
"""

import time
import asyncio
from collections import deque
from typing import Tuple, Deque


class RateLimitExceededException(Exception):
    """Raised when minute-level RPM or TPM limit is breached. Transient; retry-eligible."""
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = max(0.1, round(retry_after, 2))


class DailyBudgetExhaustedException(Exception):
    """Raised when daily request or token allocation is exhausted. Permanent; not retry-eligible."""
    def __init__(self, message: str):
        super().__init__(message)


class RateLimiter:
    def __init__(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 6000,
        daily_request_budget: int = 1000,
        daily_token_budget: int = 100000,
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.daily_request_budget = daily_request_budget
        self.daily_token_budget = daily_token_budget

        self._lock = asyncio.Lock()

        # Sliding window deques for the last 60 seconds: store (timestamp, token_count)
        self._minute_window: Deque[Tuple[float, int]] = deque()

        # Daily cumulative counters
        self._daily_requests = 0
        self._daily_tokens = 0
        self._last_day = time.strftime("%Y-%m-%d")

    def _reset_daily_if_new_day(self, current_day: str):
        if current_day != self._last_day:
            self._daily_requests = 0
            self._daily_tokens = 0
            self._last_day = current_day

    async def check_and_acquire(self, estimated_tokens: int = 400) -> float:
        """Atomically checks daily budget and sliding minute quota before admitting a request.

        Returns queue or limiter latency in seconds.
        """
        async with self._lock:
            now = time.time()
            current_day = time.strftime("%Y-%m-%d")
            self._reset_daily_if_new_day(current_day)

            # 1. Permanent Daily Budget Checks
            if self._daily_requests >= self.daily_request_budget:
                raise DailyBudgetExhaustedException(
                    f"Daily request budget exhausted: {self._daily_requests}/{self.daily_request_budget}"
                )

            if self._daily_tokens + estimated_tokens > self.daily_token_budget:
                raise DailyBudgetExhaustedException(
                    f"Daily token budget exhausted: {self._daily_tokens + estimated_tokens}/{self.daily_token_budget}"
                )

            # 2. Evict records older than 60.0 seconds from the sliding window
            cutoff = now - 60.0
            while self._minute_window and self._minute_window[0][0] <= cutoff:
                self._minute_window.popleft()

            # 3. Minute-level RPM Check
            current_rpm = len(self._minute_window)
            if current_rpm >= self.rpm_limit:
                oldest_ts = self._minute_window[0][0]
                retry_after = (oldest_ts + 60.0) - now
                raise RateLimitExceededException(
                    f"Rate limit exceeded: {current_rpm}/{self.rpm_limit} RPM",
                    retry_after=retry_after,
                )

            # 4. Minute-level TPM Check
            current_tpm = sum(entry[1] for entry in self._minute_window)
            if current_tpm + estimated_tokens > self.tpm_limit:
                # Find oldest entry contributing to breach
                oldest_ts = self._minute_window[0][0]
                retry_after = (oldest_ts + 60.0) - now
                raise RateLimitExceededException(
                    f"Token rate limit exceeded: {current_tpm + estimated_tokens}/{self.tpm_limit} TPM",
                    retry_after=retry_after,
                )

            # 5. Atomic admission
            self._minute_window.append((now, estimated_tokens))
            self._daily_requests += 1
            self._daily_tokens += estimated_tokens
            return 0.0

    async def update_actual_tokens(self, estimated_tokens: int, actual_tokens: int):
        """Reconcile difference between estimated and actual consumed tokens."""
        diff = actual_tokens - estimated_tokens
        if diff != 0:
            async with self._lock:
                self._daily_tokens += diff
                if self._minute_window:
                    # Update latest entry if still within window
                    last_ts, _ = self._minute_window[-1]
                    self._minute_window[-1] = (last_ts, max(1, actual_tokens))

    def get_status(self) -> dict:
        """Returns snapshot of current limiter state."""
        now = time.time()
        cutoff = now - 60.0
        active = [w for w in self._minute_window if w[0] > cutoff]
        return {
            "current_rpm": len(active),
            "rpm_limit": self.rpm_limit,
            "current_tpm": sum(w[1] for w in active),
            "tpm_limit": self.tpm_limit,
            "daily_requests": self._daily_requests,
            "daily_request_budget": self.daily_request_budget,
            "daily_tokens": self._daily_tokens,
            "daily_token_budget": self.daily_token_budget,
        }

