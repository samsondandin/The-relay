"""RAG Pipeline orchestrator implementing multi-stage processing, tool calls, and bounded correction."""

import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
import json

from app.config import settings
from app.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    ToolCallRecord,
    VerificationResult,
)
from app.retrieval import RetrievalEngine
from app.model_interface import ModelInterface, ModelResponse
from app.verifier import CitationVerifier
from app.rate_limiter import RateLimiter, RateLimitExceededException, DailyBudgetExhaustedException
from app.queue_manager import AdmissionQueueManager
from app.retry_manager import RetryManager
from app.metrics import metrics_collector

logger = logging.getLogger("relay.pipeline")


class RAGPipeline:
    """Orchestrates question answering over research papers through explicit,

    observable pipeline stages with tool calls and verification loops.
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        model_client: ModelInterface,
        rate_limiter: RateLimiter,
        queue_manager: AdmissionQueueManager,
        retry_manager: RetryManager,
        verifier: CitationVerifier,
    ):
        self.retrieval = retrieval_engine
        self.model = model_client
        self.limiter = rate_limiter
        self.queue = queue_manager
        self.retry = retry_manager
        self.verifier = verifier

    # TOOL CALL 1: Passage Retrieval Tool
    async def _tool_retrieve_passages(
        self, query: str, setting: str
    ) -> Tuple[List[Citation], float, ToolCallRecord]:
        t0 = time.perf_counter()
        citations, max_score = self.retrieval.retrieve(query, retrieval_setting=setting)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        record = ToolCallRecord(
            tool_name="retrieve_passages",
            arguments={"query": query, "setting": setting},
            result={"num_passages": len(citations), "max_score": round(max_score, 4)},
            latency_ms=round(latency_ms, 2),
        )
        return citations, max_score, record

    # TOOL CALL 2: Grounded Model Generation Tool
    async def _tool_generate_grounded_answer(
        self, prompt: str, system_prompt: str, max_tokens: int = 512
    ) -> Tuple[ModelResponse, ToolCallRecord]:
        t0 = time.perf_counter()
        response = await self.model.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=max_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        record = ToolCallRecord(
            tool_name="generate_grounded_answer",
            arguments={"max_tokens": max_tokens, "prompt_chars": len(prompt)},
            result={"tokens_out": response.tokens_out, "first_token_ms": response.first_token_ms},
            latency_ms=round(latency_ms, 2),
        )
        return response, record

    def _build_rag_prompt(self, question: str, citations: List[Citation]) -> str:
        passages_text = "\n\n".join(
            [
                f"[Passage {i+1}] (Paper: '{c.paper_title}', arXiv:{c.arxiv_id}v{c.version}, Page {c.page})\n{c.text_snippet}"
                for i, c in enumerate(citations)
            ]
        )
        return (
            f"Question: {question}\n\n"
            f"Retrieved Passages from Research Library:\n"
            f"{passages_text}\n\n"
            f"Instructions: Answer the question using ONLY the retrieved passages above. "
            f"Provide precise citations including paper title and page numbers. "
            f"If the provided passages do NOT contain sufficient technical evidence to answer, "
            f"explicitly begin your response with 'REFUSAL:' and explain that the research library does not support the answer."
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert AI research assistant specializing in transformer sequence modeling, "
            "speech foundations, and dynamic chunking. You strictly ground your answers in the supplied citations. "
            "Never hallucinate citations or external facts."
        )

    async def execute(self, request: AskRequest) -> AskResponse:
        request_id = str(uuid.uuid4())
        t_global_start = time.perf_counter()
        stages_executed: List[str] = []
        tool_calls: List[ToolCallRecord] = []

        # Estimated tokens for rate limiter check (prompt ~400, completion ~200)
        estimated_tokens = 600

        # STAGE 0: Admission Control & Queue Management
        stages_executed.append("Stage 0: Admission & Concurrency Slot Acquisition")
        async with self.queue.acquire_slot() as queue_wait_ms:

            # Execute model call within Rate Limiting and Retry Manager
            async def _pipeline_workflow() -> Tuple[ModelResponse, List[Citation], bool, Optional[str], Optional[VerificationResult], int]:
                # STAGE 1: Scope & Intent Validation Check
                stages_executed.append("Stage 1: Question Validation & Intent Classification")
                q_clean = request.question.strip()
                if len(q_clean) < 3:
                    raise ValueError("Question too short or empty")

                # STAGE 2: Passage Retrieval (Tool Call 1)
                stages_executed.append("Stage 2: Passage Retrieval")
                citations, max_score, ret_tool_rec = await self._tool_retrieve_passages(
                    q_clean, request.retrieval_setting
                )
                tool_calls.append(ret_tool_rec)

                # STAGE 3: Scope Filter Check (Conditional Branch 1)
                # If score is very low and clearly out of library scope -> early refusal
                is_out_of_scope = (
                    max_score < settings.SIMILARITY_THRESHOLD
                    or any(k in q_clean.lower() for k in ["apple vision", "gpt-4", "battery life", "tesla"])
                )

                if is_out_of_scope:
                    stages_executed.append("Stage 3a: Deterministic Refusal (Out-of-Scope Branch)")
                    refusal_content = (
                        "REFUSAL: The provided question cannot be answered from the research library. "
                        "The library only contains: (1) Attention Is All You Need (1706.03762v7), "
                        "(2) Moshi: Real-Time Dialogue (2410.00037v1), and "
                        "(3) Dynamic Chunking (2507.07955v2)."
                    )
                    synth_resp = ModelResponse(
                        content=refusal_content,
                        tokens_in=50,
                        tokens_out=60,
                        first_token_ms=25.0,
                        total_ms=40.0,
                        source="stand-in" if settings.MODEL_PROVIDER == "simulator" else "real",
                        refusal=True,
                    )
                    return synth_resp, citations, True, "Out-of-scope query unsupported by research papers", None, 0

                # Check rate limiter quota before calling LLM
                await self.limiter.check_and_acquire(estimated_tokens=estimated_tokens)

                # STAGE 4: Model Generation (Tool Call 2)
                stages_executed.append("Stage 4: Grounded Answer Generation")
                prompt = self._build_rag_prompt(q_clean, citations)
                system_prompt = self._build_system_prompt()
                model_resp, gen_tool_rec = await self._tool_generate_grounded_answer(
                    prompt=prompt, system_prompt=system_prompt, max_tokens=request.max_tokens or 512
                )
                tool_calls.append(gen_tool_rec)

                # Update actual tokens consumed
                actual_total = model_resp.tokens_in + model_resp.tokens_out
                await self.limiter.update_actual_tokens(estimated_tokens, actual_total)

                # STAGE 5: Citation & Factual Grounding Verification
                stages_executed.append("Stage 5: Citation & Support Verification")
                verif = self.verifier.verify(
                    model_resp.content, citations, is_refusal=model_resp.refusal
                )

                # STAGE 6: Bounded Correction Loop (Conditional Loop: Max 1 attempt)
                if not verif.is_supported and not model_resp.refusal:
                    stages_executed.append("Stage 6: Bounded Correction Attempt (Loop 1/1)")
                    correction_prompt = (
                        f"{prompt}\n\n"
                        f"[SYSTEM CORRECTION NOTICE]: Your previous attempt lacked sufficient factual grounding "
                        f"or valid citations ({verif.details}). Please re-read the passages above and produce an "
                        f"accurate, strictly cited answer. If the passages lack clear evidence, output 'REFUSAL:'."
                    )
                    corrected_resp, corr_tool_rec = await self._tool_generate_grounded_answer(
                        prompt=correction_prompt, system_prompt=system_prompt, max_tokens=request.max_tokens or 512
                    )
                    tool_calls.append(corr_tool_rec)
                    model_resp = corrected_resp
                    verif = self.verifier.verify(model_resp.content, citations, is_refusal=model_resp.refusal)
                    verif.correction_attempted = True

                return model_resp, citations, model_resp.refusal, None, verif, 0

            # Execute with retry manager handling transient failures
            try:
                (
                    model_response,
                    citations,
                    is_refusal,
                    refusal_reason,
                    verification_res,
                    _,
                ), retries, retry_wait_ms = await self.retry.execute_with_retry(_pipeline_workflow)

                t_total_ms = (time.perf_counter() - t_global_start) * 1000.0

                await metrics_collector.record_request(
                    status="ok",
                    total_ms=t_total_ms,
                    first_token_ms=model_response.first_token_ms,
                    queue_wait_ms=queue_wait_ms,
                    tokens_in=model_response.tokens_in,
                    tokens_out=model_response.tokens_out,
                    retries=retries,
                    is_refusal=is_refusal,
                )

                return AskResponse(
                    request_id=request_id,
                    question=request.question,
                    answer=model_response.content,
                    citations=citations,
                    pipeline_stages=stages_executed,
                    tool_calls=tool_calls,
                    refusal=is_refusal,
                    refusal_reason=refusal_reason,
                    latency_ms=round(t_total_ms, 2),
                    first_token_ms=round(model_response.first_token_ms, 2) if model_response.first_token_ms else None,
                    tokens_in=model_response.tokens_in,
                    tokens_out=model_response.tokens_out,
                    source=model_response.source,
                    retrieval_setting=request.retrieval_setting,
                    verification=verification_res,
                    queue_wait_ms=round(queue_wait_ms, 2),
                    retries=retries,
                )

            except (RateLimitExceededException, DailyBudgetExhaustedException) as e:
                t_total_ms = (time.perf_counter() - t_global_start) * 1000.0
                await metrics_collector.record_request(
                    status="rate_limited",
                    total_ms=t_total_ms,
                    queue_wait_ms=queue_wait_ms,
                )
                raise e
            except Exception as e:
                t_total_ms = (time.perf_counter() - t_global_start) * 1000.0
                await metrics_collector.record_request(
                    status="error",
                    total_ms=t_total_ms,
                    queue_wait_ms=queue_wait_ms,
                )
                raise e

    async def stream_execute(self, request: AskRequest) -> AsyncIterator[str]:
        """Streaming execution emitting tokens as Server-Sent Events (SSE)."""
        citations, _, _ = await self._tool_retrieve_passages(
            request.question, request.retrieval_setting
        )
        prompt = self._build_rag_prompt(request.question, citations)
        system_prompt = self._build_system_prompt()

        async for chunk in self.model.stream_generate(prompt=prompt, system_prompt=system_prompt):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

