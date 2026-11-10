# Limitations — The Relay

## 1. Citation Validation vs Factual Correctness
- **Structural Validation**: The `CitationVerifier` checks whether cited papers (arXiv IDs, titles) and page references correspond to retrieved passages, and measures n-gram lexical overlap.
- **Fundamental Limitation**: A structurally valid citation does not guarantee semantic truth. An LLM can cite *"Attention Is All You Need, page 3"* while fabricating a false numerical parameter that happens to share common vocabulary with the page. Pure syntactic or n-gram overlap cannot replace formal semantic entailment or manual ground-truth evaluation.

## 2. Lexical Vector Fallback vs Dense Cross-Encoder Embeddings
- **Implementation**: The pipeline uses scikit-learn sublinear TF-IDF with hybrid term density reranking, conforming to the assignment's explicit fallback guidance for Python 3.14.
- **Trade-off**: While fast and deterministic, TF-IDF cannot perform semantic query expansion (e.g., mapping "speech synthesis framerate" to "acoustic token stride" without shared vocabulary).

## 3. Local Rate Limiting vs Groq Cloud Enforcement
- **Local Policy**: Sliding-window log with atomic reservation.
- **Provider Reality**: Real Groq uses distributed token bucket systems across global clusters with dynamic tier leasing, variable token estimation buffers, and floating reset intervals. Our local simulator reproduces the operational impact of quotas, but does not claim to duplicate Groq's proprietary distributed rate-limiting internals.

## 4. Single-Machine Scope & Storage
- Indexing and queuing run in-process using Python async primitives (`asyncio.Semaphore`, `asyncio.Lock`, in-memory deques). For enterprise distributed scale, external backends (Redis for token-bucket rate limiting, RabbitMQ/Kafka for request queues, Milvus/Qdrant for vector indices) would be required.

