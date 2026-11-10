# Development Log — The Relay

## Phase 1: Environment Setup & Paper Acquisition
- Target platform: macOS (Darwin x86_64 / arm64), Python 3.14.6.
- Initialized isolated virtual environment `.venv`.
- Downloaded exact requested arXiv paper versions directly into `data/papers/`:
  1. Attention Is All You Need (arXiv:1706.03762v7, 15 pages)
  2. Moshi: Speech-Text Foundation Model for Real-Time Dialogue (arXiv:2410.00037v1, 66 pages)
  3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (arXiv:2507.07955v2, 39 pages)
- Verified PDF integrity using byte sizes and page extraction tests.

## Phase 2: Architecture & Ingestion Design
- Due to lack of prebuilt C-extension wheels for PyMuPDF on early Python 3.14 builds, built `DocumentProcessor` with PyMuPDF bindings and resilient `pypdf` extraction fallback.
- Extracted sections and hierarchical structural tags (Introduction, Model Architecture, Scaled Dot-Product, Dialogue Streams, Dynamic Chunking).
- Implemented configurable chunking:
  - Configuration A: 300-token chunks with 50-token overlap.
  - Configuration B: 600-token chunks with 100-token overlap.
  - Configuration C: 600-token chunks with 100-token overlap and hybrid reranker.
- Built persistent disk cache (`chunks_A.json`, `chunks_B.json`, `chunks_C.json`, `tfidf_matrix_*.pkl`, `vectorizer_*.pkl`) under `data/processed/`, enabling near-instantaneous (13ms) index reload times.

## Phase 3: RAG Pipeline & Multi-Stage Orchestration
- Implemented explicit 6-stage pipeline:
  - Stage 0: Admission control & queue wait slot acquisition.
  - Stage 1: Question validation & query intent classification.
  - Stage 2: Passage retrieval tool call (`retrieve_passages`).
  - Stage 3: Scope check conditional branch (deterministic refusal for out-of-scope or sub-threshold queries).
  - Stage 4: Grounded generation tool call (`generate_grounded_answer`).
  - Stage 5: Citation & support verification (`CitationVerifier`).
  - Stage 6: Bounded correction loop (maximum 1 correction retry if citation or factual support is weak).
- Implemented shared `ModelInterface` contract implemented by `GroqClient` and `ModelSimulator`.

## Phase 4: Admission Control, Sliding-Window Limiter & Retry Policy
- Created `RateLimiter`: sliding-window deque for atomic RPM and TPM tracking, daily budget caps, preventing race conditions via asyncio lock.
- Created `AdmissionQueueManager`: FIFO queue with bounded semaphore concurrency (`MAX_CONCURRENCY=5`), bounded waiting queue capacity (`QUEUE_CAPACITY=20`), and queue wait time measurement.
- Created `RetryManager`: exponential backoff with full jitter, honoring `Retry-After` headers, strictly excluding non-retryable permanent quota exhaustion (DailyBudgetExhaustedException).

## Phase 5: Testing, Benchmark Load Generation & Analysis
- Developed 20 automated tests in `tests/` covering retrieval, pipeline stages, sliding-window rate limiting, simulator determinism, metrics percentiles, and retry storm congestion.
- Executed full 5-scenario benchmark suite generating 159 rows in `results/results.csv`.
- Built Pandas analysis engine in `benchmarks/analyze_results.py` generating Tables 1, 2, and 3 in `report/tables.md`.

