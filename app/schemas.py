"""Pydantic schemas for request/response validation and internal data models."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    paper_title: str
    arxiv_id: str
    version: str
    page: int
    section: Optional[str] = None
    score: float = Field(default=0.0, description="Relevance score from retriever")
    text_snippet: str = Field(..., description="Excerpt from the source document")


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    latency_ms: float = 0.0


class VerificationResult(BaseModel):
    is_supported: bool
    citation_valid: bool
    overlap_score: float
    correction_attempted: bool = False
    details: str = ""


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question to answer")
    retrieval_setting: Literal["A", "B", "C"] = Field(
        default="B",
        description="Retrieval configuration: A (300tok/top3), B (600tok/top5), C (600tok/top5+rerank)"
    )
    stream: bool = Field(default=False, description="Whether to stream response tokens")
    max_tokens: Optional[int] = Field(default=None, description="Max generation tokens")


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    pipeline_stages: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    refusal: bool = False
    refusal_reason: Optional[str] = None
    latency_ms: float
    first_token_ms: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    source: Literal["real", "stand-in"]
    retrieval_setting: Literal["A", "B", "C"]
    verification: Optional[VerificationResult] = None
    queue_wait_ms: float = 0.0
    retries: int = 0


class HealthResponse(BaseModel):
    status: str
    model_provider: str
    library_loaded: bool
    total_documents: int
    total_passages: int
    retrieval_engine: str


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    rate_limited_requests: int
    failed_requests: int
    refused_requests: int
    current_queue_length: int
    active_concurrency: int
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]
    p50_first_token_ms: Optional[float]
    p95_first_token_ms: Optional[float]
    total_tokens_consumed: int
    daily_request_budget_remaining: int
    daily_token_budget_remaining: int


class BenchmarkRecord(BaseModel):
    request_id: str
    scenario: str
    source: Literal["real", "stand-in"]
    start_time: str
    first_token_ms: Optional[float] = None
    total_ms: float
    status: Literal["ok", "rate_limited", "timeout", "error"]
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retrieval_setting: Literal["A", "B", "C"]
    answer_grade: Optional[Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]] = None
    queue_wait_ms: float = 0.0
    working_ms: float = 0.0
    retry_wait_ms: float = 0.0
    http_status: int = 200
    queue_length: int = 0
    concurrency: int = 0
    cost_usd: float = 0.0
    configuration_id: str = "B"

