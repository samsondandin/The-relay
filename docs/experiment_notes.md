# Experiment Notes — The Relay

## 1. Retrieval Experiments Summary
Evaluated 15 curated questions across the three target papers across configurations A, B, and C:
- **Configuration A (300-token chunks, top 3, no reranking)**:
  - Accuracy: 86.7% correct and cited, 13.3% correct refusal.
  - Latency: TTFT p50 = 92.2 ms, Total p50 = 325.5 ms.
  - Average completion tokens: 47.3.
  - Observation: Faster prompt processing due to smaller context window (approx 420 prompt tokens). Occasionally suffers from chunk boundary truncation where architectural parameters span two chunks.
- **Configuration B (600-token chunks, top 5, no reranking)**:
  - Accuracy: 86.7% correct and cited, 13.3% correct refusal.
  - Latency: TTFT p50 = 97.0 ms, Total p50 = 286.0 ms.
  - Average completion tokens: 48.3.
  - Observation: Best overall balance of context richness and latency. Top-5 passages offer robust contextual grounding without overflowing attention context.
- **Configuration C (600-token chunks, top 5, hybrid reranked)**:
  - Accuracy: 86.7% correct and cited, 13.3% correct refusal.
  - Latency: TTFT p50 = 106.6 ms, Total p50 = 333.4 ms.
  - Average completion tokens: 46.5.
  - Observation: Adds approx 10-15ms reranking compute over candidate pool, but improves ranking confidence and exact-term keyword overlap.

## 2. Benchmark Scenarios Summary (from results/results.csv)
- **Scenario 1: Steady Traffic (N=20)**:
  - Offered rate: ~10 req/s with spacing.
  - Success rate: 65.0% success, 35.0% rate limited (due to 6000 TPM limit filling after ~13 requests).
  - Total latency: p50 = 303.8 ms, p95 = 3015.0 ms.
- **Scenario 2: Sudden Burst (N=24)**:
  - Offered rate: instantaneous burst of 12 concurrent requests.
  - Success rate: 54.2% success, 45.8% rate limited.
  - Queue behavior: Admission queue absorbed bursts up to `MAX_CONCURRENCY=5`, queuing excess requests. Peak latency reached 6577.8 ms (p95).
- **Scenario 3: Sustained Overload (N=35)**:
  - Offered rate: continuous rapid fire.
  - Success rate: 17.1% success, 82.9% rate limited.
  - Queue behavior: Token budget and RPM limits rapidly saturated. Limiter returned 429 with computed `Retry-After`.
- **Scenario 4: RPM Halved (N=20, RPM=15)**:
  - Success rate: 15.0% success, 85.0% rate limited.
  - Clear demonstration of throughput halving under provider throttling.
- **Scenario 5: Daily Budget Exhausted (N=15)**:
  - Success rate: 20.0% (initial requests before cap), 80.0% rate limited.
  - Retries/Req: exactly 0.00. Proves that the retry policy cleanly refused to retry permanent budget exhaustion.

