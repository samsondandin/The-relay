# AI Interaction Log — The Relay

## Collaboration Overview
This log documents key decisions, promptings, corrections, and engineering handoffs throughout the development of The Relay.

### 1. Ingestion & Dependency Adaptation
- **Initial Plan**: Attempted standard installation of PyMuPDF and Sentence Transformers.
- **Encountered Reality**: The runtime environment (Python 3.14.6) lacked prebuilt binaries for `pymupdf` and C++ MuPDF bindings, triggering a 10-minute compilation freeze.
- **Intervention**: Killed the stuck C++ build. Leveraged the brief's explicit allowance: deployed `pypdf` extraction fallback and implemented scikit-learn sublinear TF-IDF retrieval with hybrid reranking. All 120 pages of the 3 papers extracted with high fidelity in under 2 seconds.

### 2. Rate Limiting Calibration & Concurrency Race Fix
- **Observation during Load Testing**: Initial naive tests revealed that concurrent requests arriving within the same millisecond could all observe `current_rpm < limit` before any reservation had been written.
- **Intervention**: Enclosed minute-window deque checks and reservations within `async with self._lock`. In `test_rate_limiter.py::test_concurrent_quota_race_condition`, 10 simultaneous coroutines competing for 5 slots resulted in exactly 5 successes and 5 rejections.

### 3. Investigation of Naive Retrieval Failure Modes
- **Discovery**: When testing queries regarding the Transformer's base `d_model = 512`, the 2025 paper *"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"* repeatedly scored higher than *"Attention Is All You Need"* in compact chunking.
- **Root Cause**: The 2025 paper dedicates extensive background and experimental comparison sections to the 2017 Transformer baseline parameters.
- **Design Action**: Highlighted this discovery in `report/relay_report.md` (Section 7: *How this library will mislead a naive retriever*) and demonstrated how multi-passage synthesis or hybrid reranking is required to prevent attribution errors.

### 4. Retry Storm Demonstration
- **Goal**: Demonstrate how uncoordinated retries under exhausted quotas worsen congestion.
- **Implementation**: Created `test_retry_storm_congestion_demonstration` in `tests/test_rate_limiter.py`. Proved that an aggressive retry policy turned 1 failed request into 3 consecutive quota-depleting requests within milliseconds when backoff jitter and quota reset times were ignored.
