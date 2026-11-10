"""FastAPI main application exposing /ask, /health, and /metrics endpoints."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetricsResponse,
)
from app.retrieval import RetrievalEngine
from app.model_interface import ModelInterface
from app.groq_client import GroqClient
from app.simulator import ModelSimulator
from app.rate_limiter import (
    RateLimiter,
    RateLimitExceededException,
    DailyBudgetExhaustedException,
)
from app.queue_manager import (
    AdmissionQueueManager,
    QueueFullException,
    QueueTimeoutException,
)
from app.retry_manager import RetryManager
from app.verifier import CitationVerifier
from app.pipeline import RAGPipeline
from app.metrics import metrics_collector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("relay.main")

# State containers initialized at startup
state: dict = {}


def init_app_state():
    """Initializes retrieval engine, model client, rate limiter, queue manager, and pipeline."""
    if "pipeline" in state:
        return

    logger.info("Initializing The Relay Research Assistant state...")
    retrieval_engine = RetrievalEngine(settings.PAPERS_DIR, settings.PROCESSED_DIR)
    retrieval_engine.initialize()
    state["retrieval"] = retrieval_engine

    if settings.MODEL_PROVIDER == "groq" and settings.GROQ_API_KEY:
        logger.info(f"Using hosted Groq client with model: {settings.GROQ_MODEL}")
        model_client: ModelInterface = GroqClient(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
        )
    else:
        logger.info("Using local stand-in ModelSimulator")
        model_client = ModelSimulator(
            base_latency_ms=280.0,
            latency_jitter_ms=60.0,
            failure_probability=0.0,
            load_slowdown_factor=0.05,
            model_name=f"simulated-{settings.GROQ_MODEL}",
        )
    state["model"] = model_client

    rate_limiter = RateLimiter(
        rpm_limit=settings.RATE_LIMIT_RPM,
        tpm_limit=settings.RATE_LIMIT_TPM,
        daily_request_budget=settings.DAILY_REQUEST_BUDGET,
        daily_token_budget=settings.DAILY_TOKEN_BUDGET,
    )
    queue_manager = AdmissionQueueManager(
        max_concurrency=settings.MAX_CONCURRENCY,
        queue_capacity=settings.QUEUE_CAPACITY,
        queue_timeout_seconds=settings.QUEUE_TIMEOUT_SECONDS,
    )
    retry_manager = RetryManager(
        max_retries=settings.MAX_RETRIES,
        base_backoff_seconds=settings.RETRY_BASE_BACKOFF_SECONDS,
        max_backoff_seconds=settings.RETRY_MAX_BACKOFF_SECONDS,
    )
    verifier = CitationVerifier(min_overlap_ratio=settings.MIN_CITATION_OVERLAP_RATIO)

    pipeline = RAGPipeline(
        retrieval_engine=retrieval_engine,
        model_client=model_client,
        rate_limiter=rate_limiter,
        queue_manager=queue_manager,
        retry_manager=retry_manager,
        verifier=verifier,
    )

    state["rate_limiter"] = rate_limiter
    state["queue_manager"] = queue_manager
    state["pipeline"] = pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_app_state()
    yield
    logger.info("Shutting down The Relay application.")


app = FastAPI(
    title="The Relay - Research Assistant",
    description="Multi-stage RAG service answering technical queries with verifiable citations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceededException)
async def handle_rate_limit(request: Request, exc: RateLimitExceededException):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc), "retry_after": exc.retry_after, "type": "minute_rate_limit"},
        headers={"Retry-After": str(int(exc.retry_after))},
    )


@app.exception_handler(DailyBudgetExhaustedException)
async def handle_daily_budget(request: Request, exc: DailyBudgetExhaustedException):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc), "type": "daily_budget_exhausted"},
    )


@app.exception_handler(QueueFullException)
async def handle_queue_full(request: Request, exc: QueueFullException):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc), "type": "queue_full"},
        headers={"Retry-After": "2"},
    )


@app.exception_handler(QueueTimeoutException)
async def handle_queue_timeout(request: Request, exc: QueueTimeoutException):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": str(exc), "type": "queue_timeout"},
    )


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(payload: AskRequest):
    """Answers engineering questions using verifiable research library citations."""
    init_app_state()
    pipeline: RAGPipeline = state["pipeline"]

    if payload.stream:
        return StreamingResponse(
            pipeline.stream_execute(payload),
            media_type="text/event-stream",
        )

    response = await pipeline.execute(payload)
    return response


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health status and corpus metadata."""
    init_app_state()
    retrieval: RetrievalEngine = state.get("retrieval")
    stats = retrieval.get_corpus_stats() if retrieval else {}
    return HealthResponse(
        status="healthy",
        model_provider=settings.MODEL_PROVIDER,
        library_loaded=retrieval.is_initialized if retrieval else False,
        total_documents=stats.get("total_documents", 0),
        total_passages=stats.get("config_B_passages", 0),
        retrieval_engine=stats.get("retrieval_engine", "uninitialized"),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint():
    """Real-time observability and percentiles."""
    init_app_state()
    qm: AdmissionQueueManager = state.get("queue_manager")
    rl: RateLimiter = state.get("rate_limiter")
    rl_status = rl.get_status() if rl else {}

    summary = metrics_collector.get_summary(
        current_queue_len=qm.current_waiting if qm else 0,
        active_concurrency=qm.active_concurrency if qm else 0,
        daily_req_rem=max(0, rl.daily_request_budget - rl._daily_requests) if rl else 0,
        daily_tok_rem=max(0, rl.daily_token_budget - rl._daily_tokens) if rl else 0,
    )
    return MetricsResponse(**summary)
