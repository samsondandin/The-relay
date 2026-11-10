"""Metrics collection and aggregation module for real-time observability."""

import time
import math
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np


class MetricsAggregator:
    """Thread-safe telemetry store capturing latency distributions, token usage,

    queue states, and error rates.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limited_requests = 0
        self.failed_requests = 0
        self.refused_requests = 0
        self.total_retries = 0

        self.latencies: List[float] = []
        self.first_token_latencies: List[float] = []
        self.queue_waits: List[float] = []

        self.tokens_in_total = 0
        self.tokens_out_total = 0

    async def record_request(
        self,
        status: str,
        total_ms: float,
        first_token_ms: Optional[float] = None,
        queue_wait_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        retries: int = 0,
        is_refusal: bool = False,
    ):
        async with self._lock:
            self.total_requests += 1
            self.total_retries += retries
            self.queue_waits.append(queue_wait_ms)

            if status == "ok":
                self.successful_requests += 1
                self.latencies.append(total_ms)
                if first_token_ms is not None:
                    self.first_token_latencies.append(first_token_ms)
                self.tokens_in_total += tokens_in
                self.tokens_out_total += tokens_out
                if is_refusal:
                    self.refused_requests += 1
            elif status == "rate_limited":
                self.rate_limited_requests += 1
            else:
                self.failed_requests += 1

    def _calc_percentile(self, values: List[float], p: float) -> Optional[float]:
        if not values:
            return None
        return round(float(np.percentile(values, p)), 2)

    def get_summary(
        self,
        current_queue_len: int = 0,
        active_concurrency: int = 0,
        daily_req_rem: int = 1000,
        daily_tok_rem: int = 100000,
    ) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "failed_requests": self.failed_requests,
            "refused_requests": self.refused_requests,
            "total_retries": self.total_retries,
            "current_queue_length": current_queue_len,
            "active_concurrency": active_concurrency,
            "p50_latency_ms": self._calc_percentile(self.latencies, 50),
            "p95_latency_ms": self._calc_percentile(self.latencies, 95),
            "p99_latency_ms": self._calc_percentile(self.latencies, 99),
            "p50_first_token_ms": self._calc_percentile(self.first_token_latencies, 50),
            "p95_first_token_ms": self._calc_percentile(self.first_token_latencies, 95),
            "p50_queue_wait_ms": self._calc_percentile(self.queue_waits, 50),
            "p95_queue_wait_ms": self._calc_percentile(self.queue_waits, 95),
            "total_tokens_consumed": self.tokens_in_total + self.tokens_out_total,
            "daily_request_budget_remaining": daily_req_rem,
            "daily_token_budget_remaining": daily_tok_rem,
        }


metrics_collector = MetricsAggregator()

