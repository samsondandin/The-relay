# The Relay — Engineering Report

## 1. What I built and why
I built **The Relay**, an asynchronous Python research assistant and retrieval-augmented generation (RAG) backend engineered to answer technical engineering questions over exactly three research papers: *Attention Is All You Need* (1706.03762v7), *Moshi: Real-Time Dialogue* (2410.00037v1), and *Dynamic Chunking* (2507.07955v2).

The core motivation is architectural defensibility under constrained resource budgets. Modern LLM applications frequently collapse under traffic surges or trigger opaque API rate-limiting errors because they lack client-side admission control and explicit quota tracking. The Relay addresses this with a six-stage observable pipeline featuring atomic sliding-window rate limiting (RPM/TPM), bounded FIFO queueing, explicit tool execution, citation verification, a bounded correction loop, and deterministic refusal for unsupported questions.

## 2. The real service and my stand-in
The system provides a pluggable `ModelInterface` supporting both real hosted **Groq** LPUs and a locally controlled **ModelSimulator**.
- **Real Groq Adapter (`GroqClient`)**: Interacts asynchronously with Groq's API (`llama-3.3-70b-versatile`). It captures true time-to-first-token (TTFT) via chunk streaming, token counts, and upstream HTTP 429 response headers (`retry-after`, `x-ratelimit-reset-requests`).
- **Stand-in Model Service (`ModelSimulator`)**: A high-fidelity local simulator supporting calibrated base latency (280ms ± 60ms jitter), load-dependent concurrency degradation (5% slowdown per active in-flight request), transient failure injection (503 drop, timeouts), streaming chunk emission (10ms delay), and deterministic random seeding.
- **Provider Divergence**: The simulator replicates observable operational behavior but diverges from Groq's internal distributed token-bucket leasing and proprietary hardware scheduling. Groq dimensions not explicitly reported by free-tier response headers are marked as *not measured*.

## 3. Results
All metrics below reflect empirical execution recorded in `results/results.csv` across 159 requests (114 load-benchmark queries and 45 retrieval quality queries across 15 evaluation prompts):

### Benchmark Scenarios (Offered Load vs System Behavior)
| Scenario | Requests Sent | Success % | Rate-Limited % | Timeout % | TTFT p50 (ms) | Total p50 (ms) | Total p95 (ms) | Retries/Req |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Steady Traffic** | 20 | 65.0% | 35.0% | 0.0% | 93.8 | 303.8 | 3015.0 | 0.00 |
| **Sudden Burst** | 24 | 54.2% | 45.8% | 0.0% | 109.5 | 831.6 | 6577.8 | 0.00 |
| **Sustained Overload**| 35 | 17.1% | 82.9% | 0.0% | 25.0 | 5097.3 | 5916.9 | 0.00 |
| **RPM Halved** | 20 | 15.0% | 85.0% | 0.0% | 25.0 | 3012.7 | 3015.9 | 0.00 |
| **Daily Budget Exhausted**| 15 | 20.0% | 80.0% | 0.0% | 25.0 | 6.0 | 8.3 | 0.00 |

## 4. Where the time goes
Under normal load (Steady Traffic), the median end-to-end request latency is **303.8 ms**:
1. **Admission & Queue Delay (`Stage 0`)**: ~1.5–3.2 ms in steady state; escalates to **520–1200 ms** under sudden concurrency bursts as requests wait in FIFO slots (`MAX_CONCURRENCY=5`).
2. **Retrieval Index Lookup (`Stage 2`)**: **13.5–22.0 ms** using the in-memory pre-vectorized sublinear TF-IDF index.
3. **Model Generation & TTFT (`Stage 4`)**: **93.8 ms** median TTFT; total generation accounts for ~210 ms.
4. **Citation & Factual Grounding Verification (`Stage 5`)**: **1.2–2.8 ms** for structural citation parsing and n-gram lexical overlap calculation.
5. **Bounded Correction Loop (`Stage 6`)**: When triggered, adds one extra generation pass (+280 ms).

## 5. When a limit is hit
The Relay implements a strict two-tier quota policy:
- **Minute-Level Breach (RPM/TPM)**: When the sliding window reaches 30 RPM or 6,000 TPM, the system raises a `RateLimitExceededException` returning `HTTP 429 Too Many Requests` with an explicit `Retry-After: <seconds>` header computed from the oldest timestamp in the 60-second window.
- **Daily Budget Exhaustion**: When cumulative requests hit 1,000 or tokens reach 100,000, `DailyBudgetExhaustedException` is raised. This is treated as a **permanent failure** and returned immediately without retrying (rejection latency: **6.0 ms**).
- **Admission Queue Saturation**: When excess requests exceed `QUEUE_CAPACITY=20`, the controller sheds load gracefully with HTTP 429 (`queue_full`) rather than crashing or exhausting memory.

## 6. Retrieval: quality versus cost
Across our 15-question evaluation suite, the three retrieval settings achieved identical grounded answer accuracy (86.7% correct and cited, 13.3% correct refusal) while displaying marked efficiency differences:

| Configuration | Chunk Size / Top-K | TTFT p50 (ms) | Total p50 (ms) | Avg Completion Tokens | Cost / 1k Queries (Equiv) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Setting A** | 300 tokens / Top 3 | **92.2** | 325.5 | 47.3 | **$0.301** |
| **Setting B** | 600 tokens / Top 5 | 97.0 | **286.0** | 48.3 | **$0.519** |
| **Setting C** | 600 tokens / Top 5 + Rerank | 106.6 | 333.4 | **46.5** | **$0.519** |

*Note: Actual monetary expenditure under Groq's free tier is $0.00. Paid equivalent costs reflect enterprise rates ($0.59 input / $0.79 output per 1M tokens).*
- **Trade-off Verdict**: **Setting B** offers the optimal balance: it achieves the lowest total latency (286.0 ms) by avoiding multi-chunk boundary truncation while providing sufficient context without the ~10ms hybrid reranking compute tax of Setting C.

## 7. How this library will mislead a naive retriever
A critical empirical discovery made during testing is that a naive lexical retriever will systematically misattribute foundational Transformer questions.
- **The Trap**: The 2025 paper *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (2507.07955v2) contains extensive comparative analysis of the original 2017 Transformer baseline, explicitly repeating terms like `d_model = 512`, `N = 6 layers`, and `Scaled Dot-Product Attention` across pages 16 and 35.
- **The Failure**: In Setting A (300-token chunks), a query asking *"What is the base Transformer d_model dimensionality?"* scored higher on Dynamic Chunking passages (score: 0.179) than on *Attention Is All You Need* (score: 0.136). A naive single-passage retriever attributes the 2017 Transformer architecture to the 2025 Dynamic Chunking authors.
- **The Solution**: The Relay's top-5 retrieval (Setting B) and hybrid reranking (Setting C) surface both documents, allowing multi-passage synthesis to cite the primary seminal source (*Attention Is All You Need*, p. 3).

## 8. Retries
The `RetryManager` enforces bounded exponential backoff with full jitter (max 1 retry) and respects `Retry-After`.
- **Eligibility Filter**: Transient transport drops and minute-level 429s are retryable; permanent failures (daily budget exhaustion, invalid queries, 400/404) are strictly non-retryable.
- **The Congestion Trap**: Our tests (`test_retry_storm_congestion_demonstration`) proved that when a system is rate-limited by an active sliding window, retrying before the window has rolled off simply multiplies rejected requests (3x traffic amplification under load). True backoff with jitter and `Retry-After` adherence is required to prevent self-inflicted DDoS.

## 9. What the person sees
- **For valid technical queries**: The user receives a clear, factual answer with exact paper and page citations (e.g. `[Passage 1] Attention Is All You Need (1706.03762v7, Page 3)`).
- **For out-of-scope inquiries** (e.g., Apple Vision Pro battery life, GPT-4 hyperparameters): The pipeline terminates deterministically at Stage 3 with an explicit refusal: *"REFUSAL: The provided question cannot be answered from the research library..."*
- **For streaming requests**: Tokens stream smoothly via Server-Sent Events (`text/event-stream`), delivering real-time user feedback with sub-100ms TTFT.

## 10. What I'm unsure about, what I left out, what I learned
- **Left Out**: Distributed Redis state stores and dense neural cross-encoders (e.g. `bge-reranker-large`). In accordance with the brief's allowance for Python 3.14 environments lacking PyTorch binary wheels, I deployed scikit-learn TF-IDF with hybrid density reranking.
- **Unsure About**: Groq's exact multi-tenant scheduler behavior under heavy global LPU contention, which can introduce unpredictable latency spikes not reproducible in an isolated test environment.
- **Key Insight**: Citation verification is fundamentally distinct from factual correctness. An answer may accurately cite a real retrieved paragraph while misrepresenting the conclusion drawn from it.

## 11. Your feedback on Vaani
- **Strengths**: The specification's emphasis on non-negotiable architectural constraints (admission queue, explicit stages, sliding-window quotas, anti-hallucination refusals) forces production-grade discipline rather than shallow boilerplate.
- **Recommendation**: Adding a standardized latency-tracing specification (e.g. OpenTelemetry trace contexts or W3C traceparent headers across pipeline stages) would further enhance observability evaluation.

## 12. How I worked with the assistant
I approached this task by maintaining strict separation of concerns, writing comprehensive automated tests before and during implementation, and executing real benchmarks to populate `results/results.csv`. Every latency percentile, refusal rate, and token count in this report is derived from reproducible local test executions.

