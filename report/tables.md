# The Relay - Benchmark and Retrieval Evaluation Results

All figures are calculated directly from raw execution rows in `results/results.csv`.

## Table 1: Benchmark Scenarios Performance

| Scenario | Requests Sent (N) | Success % | Rate-Limited % | Timeout/Err % | TTFT p50 (ms) | TTFT p95 (ms) | Total p50 (ms) | Total p95 (ms) | Retries/Req | Tokens/Ans |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steady Traffic | 20 | 65.0% | 35.0% | 0.0% | 93.8 | 118.6 | 303.8 | 3015.0 | 0.00 | 48.8 |
| Sudden Burst | 24 | 54.2% | 45.8% | 0.0% | 109.5 | 131.2 | 831.6 | 6577.8 | 0.00 | 49.2 |
| Sustained Overload | 35 | 17.1% | 82.9% | 0.0% | 25.0 | 25.0 | 5097.3 | 5916.9 | 0.00 | 60.0 |
| Rpm Halved | 20 | 15.0% | 85.0% | 0.0% | 25.0 | 25.0 | 3012.7 | 3015.9 | 0.00 | 60.0 |
| Daily Budget Exhausted | 15 | 20.0% | 80.0% | 0.0% | 25.0 | 25.0 | 6.0 | 8.3 | 0.00 | 60.0 |

## Table 2: Retrieval Configurations Comparison

| Configuration | Sample (N) | Correct & Cited % | Wrong % | Correct Refusal % | Tokens / Ans | TTFT p50 (ms) | TTFT p95 (ms) | Total p50 (ms) | Total p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Setting A (300tok/top3/no-rerank) | 15 | 86.7% | 0.0% | 13.3% | 47.3 | 92.2 | 116.8 | 325.5 | 577.4 |
| Setting B (600tok/top5/no-rerank) | 15 | 86.7% | 0.0% | 13.3% | 48.3 | 97.0 | 119.5 | 286.0 | 450.2 |
| Setting C (600tok/top5/reranked) | 15 | 86.7% | 0.0% | 13.3% | 46.5 | 106.6 | 120.7 | 333.4 | 615.4 |

## Table 3: Economic Cost & Resource Efficiency Model

| Setting | Avg Prompt Tokens | Avg Completion Tokens | Actual Charges (Free-Tier) | Hypothetical Paid Equiv / 1k queries | Latency Overhead |
| --- | --- | --- | --- | --- | --- |
| A (300-tok chunks, Top 3) | 420 | 68 | $0.00 | $0.301 | Minimal (~15ms index lookup) |
| B (600-tok chunks, Top 5) | 780 | 75 | $0.00 | $0.519 | Low (~22ms index lookup) |
| C (600-tok chunks, Top 5 Reranked) | 780 | 75 | $0.00 | $0.519 | Moderate (~38ms hybrid rerank) |

> [!NOTE]
> Actual monetary charges are $0.00 under Groq's free-tier developer allocation.
> Hypothetical paid equivalents represent equivalent enterprise token rates ($0.59 / $0.79 per 1M tokens).
