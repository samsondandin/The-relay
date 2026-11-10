# The Relay — Research Assistant RAG Application

The Relay is an asynchronous, production-grade Python research assistant and retrieval-augmented generation (RAG) pipeline built with FastAPI, PyMuPDF/PyPDF, scikit-learn, and Groq/ModelSimulator. It accepts technical engineering questions, retrieves passages from three foundational research papers, generates grounded responses with exact citations, refuses out-of-scope inquiries, and enforces admission control, sliding-window rate limiting, and bounded retry policies.

---

## 1. Quick Start

### Prerequisites
- Python 3.11+ (tested on Python 3.14.6 macOS Darwin)
- Virtual environment (`.venv`)

### Installation
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
To run against real hosted Groq models, set:
```bash
GROQ_API_KEY=gsk_your_real_key_here
MODEL_PROVIDER=groq
```
To run against the local stand-in model simulator (default):
```bash
MODEL_PROVIDER=simulator
```

---

## 2. Running The Relay

### Start the API Server
```bash
source .venv/bin/activate
python run.py serve --port 8000
```
API Documentation will be live at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Live telemetry: `http://127.0.0.1:8000/metrics`

### Ask a Technical Question
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the dimensionality of d_model in the base Transformer?",
    "retrieval_setting": "B"
  }'
```

### Ask an Out-of-Scope Question (Refusal Test)
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the battery life of Apple Vision Pro?",
    "retrieval_setting": "B"
  }'
```

---

## 3. Running Benchmarks & Tests

### Execute Pytest Suite (20 Automated Tests)
```bash
source .venv/bin/activate
PYTHONPATH=. pytest -v
```

### Run Benchmark Scenarios & Retrieval Experiments
```bash
source .venv/bin/activate
python run.py benchmarks
```
Outputs:
- Granular raw execution log: `results/results.csv` (159 recorded requests)
- Analytical Markdown tables: `report/tables.md`
- Core evaluation report: `report/relay_report.md`

---

## 4. Project Architecture

```
the-relay/
├── app/
│   ├── main.py                 # FastAPI application & lifespan
│   ├── config.py               # Pydantic settings & environment configuration
│   ├── schemas.py              # Pydantic schemas (AskRequest, Citation, BenchmarkRecord)
│   ├── pipeline.py             # 6-stage RAG pipeline with tool calls & correction loop
│   ├── retrieval.py            # RetrievalEngine (Settings A, B, C with hybrid reranker)
│   ├── document_processor.py   # PDF text extraction, section detection & chunking
│   ├── model_interface.py      # Abstract ModelInterface base class & ModelResponse
│   ├── groq_client.py          # Real Groq model adapter with TTFT and token tracking
│   ├── simulator.py            # Stand-in local ModelSimulator with jitter & load degradation
│   ├── rate_limiter.py         # Sliding-window log rate limiter (RPM & TPM) & daily budget
│   ├── queue_manager.py        # Admission controller, FIFO bounded queue & wait tracking
│   ├── retry_manager.py        # Exponential backoff with jitter & Retry-After support
│   ├── verifier.py             # Structural citation validation & factual overlap checker
│   └── metrics.py              # Real-time metrics aggregator (p50, p95, p99 latencies)
├── data/
│   ├── papers/                 # Downloaded exact arXiv PDFs (1706.03762v7, 2410.00037v1, 2507.07955v2)
│   ├── processed/              # Pre-processed chunks and TF-IDF matrices
│   └── questions.jsonl         # 15 original curated test questions with reference answers
├── benchmarks/
│   ├── run_benchmarks.py       # Orchestrates 5 load scenarios and 3 retrieval evaluations
│   ├── load_generator.py       # High performance asyncio concurrent load generator
│   └── analyze_results.py      # Pandas analytical engine generating report tables
├── simulator_service/
│   └── main.py                 # Standalone microservice mimicking Groq endpoint
├── results/
│   ├── results.csv             # Primary benchmark results file (mandatory schema)
│   └── benchmark_config.json   # Benchmark scenario configurations & cost model
├── tests/
│   ├── test_retrieval.py       # Tests chunking, indexing, and Settings A/B/C
│   ├── test_pipeline.py        # Integration tests for /ask, /health, /metrics, streaming
│   ├── test_rate_limiter.py    # Tests RPM/TPM sliding windows, race conditions & retry storms
│   ├── test_simulator.py       # Tests simulator latency, determinism, and failures
│   └── test_metrics.py         # Tests percentile calculations and telemetry
├── report/
│   ├── relay_report.md         # Final 2-page engineering report
│   └── tables.md               # Empirical results tables computed from results.csv
└── run.py                      # Unified CLI entrypoint
```

