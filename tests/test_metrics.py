"""Unit tests for telemetry, latency percentiles, and metrics aggregation."""

import pytest
from app.metrics import MetricsAggregator


@pytest.mark.asyncio
async def test_metrics_percentiles_calculation():
    agg = MetricsAggregator()

    # Record 100 requests with latencies 1..100
    for lat in range(1, 101):
        await agg.record_request(
            status="ok",
            total_ms=float(lat),
            first_token_ms=float(lat) * 0.35,
            queue_wait_ms=1.5,
            tokens_in=10,
            tokens_out=20,
            retries=0,
        )

    summary = agg.get_summary()
    assert summary["total_requests"] == 100
    assert summary["successful_requests"] == 100
    assert summary["failed_requests"] == 0
    assert summary["p50_latency_ms"] == pytest.approx(50.5, abs=1.0)
    assert summary["p95_latency_ms"] == pytest.approx(95.0, abs=1.0)
    assert summary["p99_latency_ms"] == pytest.approx(99.0, abs=1.0)
    assert summary["total_tokens_consumed"] == 3000


@pytest.mark.asyncio
async def test_metrics_error_and_refusal_tracking():
    agg = MetricsAggregator()

    await agg.record_request(status="ok", total_ms=50.0, is_refusal=True)
    await agg.record_request(status="rate_limited", total_ms=10.0)
    await agg.record_request(status="error", total_ms=15.0)

    summary = agg.get_summary()
    assert summary["total_requests"] == 3
    assert summary["successful_requests"] == 1
    assert summary["refused_requests"] == 1
    assert summary["rate_limited_requests"] == 1
    assert summary["failed_requests"] == 1

