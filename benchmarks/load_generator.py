"""Asyncio-based high performance load generator for controlled load scenarios."""

import time
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from httpx import ASGITransport

from app.schemas import BenchmarkRecord


class LoadGenerator:
    """Orchestrates concurrent load generation, tracking granular telemetry

    (TTFT, total latency, queue delays, and error classifications).
    """

    def __init__(self, app_instance, base_url: str = "http://testserver"):
        self.app = app_instance
        self.base_url = base_url

    async def execute_request(
        self,
        client: httpx.AsyncClient,
        question: str,
        scenario: str,
        retrieval_setting: str = "B",
        source: str = "stand-in",
        answer_grade: Optional[str] = None,
        timeout_sec: float = 15.0,
    ) -> BenchmarkRecord:
        req_id = f"bench-{uuid.uuid4().hex[:8]}"
        start_time_iso = datetime.utcnow().isoformat()
        t0 = time.perf_counter()

        payload = {
            "question": question,
            "retrieval_setting": retrieval_setting,
            "stream": False,
        }

        try:
            resp = await client.post(
                "/ask",
                json=payload,
                timeout=timeout_sec,
            )
            total_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                first_token_ms = data.get("first_token_ms")
                tokens_in = data.get("tokens_in", 0)
                tokens_out = data.get("tokens_out", 0)
                retries = data.get("retries", 0)
                queue_wait = data.get("queue_wait_ms", 0.0)
                working_ms = max(0.0, total_ms - queue_wait)

                # Estimated cost per 1,000 queries based on Llama-3.3-70B pricing ($0.59 / $0.79 per 1M)
                cost_usd = (tokens_in * 0.00000059) + (tokens_out * 0.00000079)

                return BenchmarkRecord(
                    request_id=req_id,
                    scenario=scenario,
                    source=source,
                    start_time=start_time_iso,
                    first_token_ms=round(first_token_ms, 2) if first_token_ms else None,
                    total_ms=round(total_ms, 2),
                    status="ok",
                    retries=retries,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    retrieval_setting=retrieval_setting,
                    answer_grade=answer_grade,
                    queue_wait_ms=round(queue_wait, 2),
                    working_ms=round(working_ms, 2),
                    retry_wait_ms=0.0,
                    http_status=resp.status_code,
                    cost_usd=round(cost_usd, 6),
                    configuration_id=retrieval_setting,
                )

            elif resp.status_code == 429:
                return BenchmarkRecord(
                    request_id=req_id,
                    scenario=scenario,
                    source=source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=round(total_ms, 2),
                    status="rate_limited",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=retrieval_setting,
                    answer_grade=None,
                    queue_wait_ms=0.0,
                    working_ms=0.0,
                    retry_wait_ms=0.0,
                    http_status=429,
                    cost_usd=0.0,
                    configuration_id=retrieval_setting,
                )

            elif resp.status_code in (504, 408):
                return BenchmarkRecord(
                    request_id=req_id,
                    scenario=scenario,
                    source=source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=round(total_ms, 2),
                    status="timeout",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=retrieval_setting,
                    answer_grade=None,
                    http_status=resp.status_code,
                    cost_usd=0.0,
                    configuration_id=retrieval_setting,
                )

            else:
                return BenchmarkRecord(
                    request_id=req_id,
                    scenario=scenario,
                    source=source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=round(total_ms, 2),
                    status="error",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=retrieval_setting,
                    answer_grade=None,
                    http_status=resp.status_code,
                    cost_usd=0.0,
                    configuration_id=retrieval_setting,
                )

        except (httpx.TimeoutException, asyncio.TimeoutError):
            total_ms = (time.perf_counter() - t0) * 1000.0
            return BenchmarkRecord(
                request_id=req_id,
                scenario=scenario,
                source=source,
                start_time=start_time_iso,
                first_token_ms=None,
                total_ms=round(total_ms, 2),
                status="timeout",
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting=retrieval_setting,
                answer_grade=None,
                http_status=504,
                cost_usd=0.0,
                configuration_id=retrieval_setting,
            )

        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return BenchmarkRecord(
                request_id=req_id,
                scenario=scenario,
                source=source,
                start_time=start_time_iso,
                first_token_ms=None,
                total_ms=round(total_ms, 2),
                status="error",
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting=retrieval_setting,
                answer_grade=None,
                http_status=500,
                cost_usd=0.0,
                configuration_id=retrieval_setting,
            )

    async def run_scenario(
        self,
        scenario_name: str,
        questions: List[str],
        num_requests: int,
        concurrency: int,
        interval_ms: int = 0,
        retrieval_setting: str = "B",
        source: str = "stand-in",
    ) -> List[BenchmarkRecord]:
        transport = ASGITransport(app=self.app)
        semaphore = asyncio.Semaphore(concurrency)
        results: List[BenchmarkRecord] = []

        async with httpx.AsyncClient(transport=transport, base_url=self.base_url) as client:
            async def worker(idx: int):
                async with semaphore:
                    if interval_ms > 0 and idx > 0:
                        await asyncio.sleep((interval_ms * idx) / 1000.0)
                    q = questions[idx % len(questions)]
                    rec = await self.execute_request(
                        client=client,
                        question=q,
                        scenario=scenario_name,
                        retrieval_setting=retrieval_setting,
                        source=source,
                    )
                    results.append(rec)

            tasks = [asyncio.create_task(worker(i)) for i in range(num_requests)]
            await asyncio.gather(*tasks)

        return results
