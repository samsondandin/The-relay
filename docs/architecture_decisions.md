# Architecture Decisions (ADR) — The Relay

## ADR 01: Scikit-Learn Sublinear TF-IDF + Hybrid Density Reranker vs FAISS/SentenceTransformers
- **Context**: The project environment is running Python 3.14.6 where FAISS-CPU and PyTorch/SentenceTransformers lacked compatible binary wheels and required heavyweight compilation from source. The assignment brief specifically authorized: *"If Sentence Transformers or FAISS causes installation problems, implement a clearly documented scikit-learn TF-IDF retrieval fallback. Do not silently claim that fallback is embedding-based retrieval."*
- **Decision**: Implemented `TfidfVectorizer` (sublinear term frequency, (1, 2) n-grams, stop-words removal) with cosine similarity for candidate retrieval, and an exact-match term density hybrid reranker for Setting C.
- **Consequences**: Fast, zero-C-extension overhead, deterministic retrieval latency (~13-25ms), and fully documented as lexical vector space retrieval rather than dense dense neural embeddings.

## ADR 02: Sliding-Window Log Rate Limiting vs Fixed Window
- **Context**: Fixed-window limiters suffer from boundary burst vulnerabilities (e.g. 30 requests at second 59 and 30 requests at second 61 equates to 60 requests in 2 seconds, violating rate limits).
- **Decision**: Implemented a Sliding-Window Log rate limiter using an in-memory double-ended queue (`deque`) protected by an `asyncio.Lock`.
- **Consequences**: Accurate 60-second window quota enforcement. Concurrency race conditions are eliminated because token and request checks and reservation occur atomically inside the lock.

## ADR 03: Separate Transient Minute Quota from Permanent Daily Budget
- **Context**: Retrying a request that failed due to permanent daily quota exhaustion generates needless load and worsens server congestion.
- **Decision**: Differentiated `RateLimitExceededException` (transient, retry eligible with computed `Retry-After`) from `DailyBudgetExhaustedException` (permanent, non-retryable, immediate HTTP 429).
- **Consequences**: When daily quota is exhausted, the retry manager terminates immediately with 0 retries.

## ADR 04: Explicit Multi-Stage Pipeline with Bounded Correction Loop
- **Context**: Heavyweight agent frameworks (e.g. LangChain, CrewAI) introduce opaque abstractions, unpredictable execution paths, and high latency overhead.
- **Decision**: Designed an explicit, observable pipeline with 6 distinct stages, instrumented tool call records, and a single bounded correction attempt (maximum 1 loop).
- **Consequences**: Complete visibility into pipeline state, reproducible execution traces, and bounded latency guarantees.

## ADR 05: Abstract Model Interface for Seamless Stand-in / Real Provider Swapping
- **Context**: The system must run either against hosted Groq LPUs or against a local stand-in model simulator without code modifications in the core application.
- **Decision**: Implemented an abstract base class `ModelInterface` with `generate` and `stream_generate` methods. `app/main.py` dynamically injects `GroqClient` when `GROQ_API_KEY` is present and `MODEL_PROVIDER=groq`, or `ModelSimulator` otherwise.
- **Consequences**: Zero change to application endpoints, pipeline logic, or telemetry when alternating between real and simulated backends.

