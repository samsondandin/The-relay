"""Retry manager implementing exponential backoff with full jitter and eligibility checks."""

import time
import random
import asyncio
import logging
from typing import Callable, Any, Tuple, Optional
from app.rate_limiter import RateLimitExceededException, DailyBudgetExhaustedException
from app.queue_manager import QueueFullException, QueueTimeoutException

logger = logging.getLogger("relay.retry_manager")


class PermanentFailureException(Exception):
    """Failure that must not be retried under any circumstances."""
    pass


class RetryManager:
    """Bounded retry orchestrator preventing retry storms and observing Retry-After contracts."""

    def __init__(
        self,
        max_retries: int = 1,
        base_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 3.0,
    ):
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    def is_retryable(self, exception: Exception) -> bool:
        """Determines whether an exception qualifies for transient retry."""
        # Non-retryable permanent conditions
        if isinstance(exception, (DailyBudgetExhaustedException, QueueFullException, ValueError)):
            return False

        # Retryable transient conditions
        if isinstance(exception, (RateLimitExceededException, QueueTimeoutException, TimeoutError, asyncio.TimeoutError)):
            return True

        # Check HTTP / SDK error status codes if available
        status_code = getattr(exception, "status_code", None)
        if status_code in (429, 500, 502, 503, 504):
            return True
        if status_code in (400, 401, 403, 404, 422):
            return False

        # Generic network or transient connection errors
        msg = str(exception).lower()
        if "rate limit" in msg or "429" in msg or "timeout" in msg or "connection reset" in msg:
            return True

        return False

    def compute_backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Computes exponential backoff with full jitter (decorrelated / truncated exponential).

        Respects Retry-After header if provided.
        """
        if retry_after is not None and retry_after > 0:
            # Respect explicit provider header with slight positive jitter
            return min(self.max_backoff_seconds, retry_after + random.uniform(0.05, 0.25))

        # Full jitter: sleep = random(0, min(max_backoff, base * 2^attempt))
        upper = min(self.max_backoff_seconds, self.base_backoff_seconds * (2 ** attempt))
        return random.uniform(0.1, upper)

    async def execute_with_retry(
        self,
        coro_func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Tuple[Any, int, float]:
        """Executes the coroutine, retrying at most max_retries times upon retryable errors.

        Returns (result, retries_taken, total_retry_wait_ms).
        """
        retries = 0
        total_wait_ms = 0.0

        while True:
            try:
                result = await coro_func(*args, **kwargs)
                return result, retries, total_wait_ms
            except Exception as e:
                if not self.is_retryable(e) or retries >= self.max_retries:
                    logger.warning(
                        f"Retry aborted or exhausted. Retries: {retries}/{self.max_retries}. Reason: {e}"
                    )
                    raise e

                retries += 1
                retry_after = getattr(e, "retry_after", None)
                wait_sec = self.compute_backoff(retries, retry_after=retry_after)
                wait_ms = wait_sec * 1000.0
                total_wait_ms += wait_ms

                logger.info(
                    f"Transient failure ({type(e).__name__}: {e}). Retrying ({retries}/{self.max_retries}) after {wait_sec:.2f}s."
                )
                await asyncio.sleep(wait_sec)

