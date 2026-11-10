"""Unit tests for RateLimiter, sliding-window enforcement, concurrency, and retry congestion."""

import time
import asyncio
import pytest
from app.rate_limiter import (
    RateLimiter,
    RateLimitExceededException,
    DailyBudgetExhaustedException,
)
from app.retry_manager import RetryManager


@pytest.mark.asyncio
async def test_rpm_limit_sliding_window():
    limiter = RateLimiter(rpm_limit=3, tpm_limit=5000, daily_request_budget=100)

    # Acquire 3 slots successfully
    await limiter.check_and_acquire(estimated_tokens=100)
    await limiter.check_and_acquire(estimated_tokens=100)
    await limiter.check_and_acquire(estimated_tokens=100)

    # 4th slot should raise RateLimitExceededException
    with pytest.raises(RateLimitExceededException) as exc_info:
        await limiter.check_and_acquire(estimated_tokens=100)

    assert exc_info.value.retry_after > 0


@pytest.mark.asyncio
async def test_tpm_limit_sliding_window():
    limiter = RateLimiter(rpm_limit=10, tpm_limit=500, daily_request_budget=100)

    # Consume 400 tokens
    await limiter.check_and_acquire(estimated_tokens=400)

    # Consuming another 200 breaches 500 TPM
    with pytest.raises(RateLimitExceededException) as exc_info:
        await limiter.check_and_acquire(estimated_tokens=200)

    assert "Token rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_daily_request_budget_exhaustion():
    limiter = RateLimiter(rpm_limit=10, tpm_limit=5000, daily_request_budget=2)

    await limiter.check_and_acquire(10)
    await limiter.check_and_acquire(10)

    # 3rd request should fail permanently
    with pytest.raises(DailyBudgetExhaustedException) as exc_info:
        await limiter.check_and_acquire(10)

    assert "Daily request budget exhausted" in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrent_quota_race_condition():
    """Ensure concurrent tasks cannot oversaturate remaining quota."""
    limiter = RateLimiter(rpm_limit=5, tpm_limit=10000, daily_request_budget=100)

    successes = 0
    failures = 0

    async def attempt():
        nonlocal successes, failures
        try:
            await limiter.check_and_acquire(50)
            successes += 1
        except RateLimitExceededException:
            failures += 1

    # Launch 10 simultaneous coroutines
    await asyncio.gather(*[attempt() for _ in range(10)])

    # Exactly 5 should succeed and 5 rejected
    assert successes == 5
    assert failures == 5


@pytest.mark.asyncio
async def test_retry_storm_congestion_demonstration():
    """Demonstrates how unconstrained or immediate retries exacerbate congestion

    under a constrained rate limit vs bounded jittered backoff.
    """
    limiter = RateLimiter(rpm_limit=4, tpm_limit=10000, daily_request_budget=100)
    retry_manager = RetryManager(max_retries=2, base_backoff_seconds=0.05, max_backoff_seconds=0.2)

    total_attempts = 0

    async def failing_under_load():
        nonlocal total_attempts
        total_attempts += 1
        await limiter.check_and_acquire(100)
        return "ok"

    # Fill quota with 4 items
    for _ in range(4):
        await limiter.check_and_acquire(100)

    # Now attempt a request with retry manager: since quota is exhausted, every retry attempt
    # repeatedly hits the barrier until retry budget is depleted
    with pytest.raises(RateLimitExceededException):
        await retry_manager.execute_with_retry(failing_under_load)

    # Initial attempt + 2 retries = 3 attempts contributed to congestion
    assert total_attempts == 3
