"""Stand-in simulated model service implementing ModelInterface for controlled load experiments."""

import time
import random
import asyncio
import logging
from typing import Optional, Dict, Any, AsyncIterator
from app.model_interface import ModelInterface, ModelResponse
from app.rate_limiter import RateLimitExceededException, DailyBudgetExhaustedException

logger = logging.getLogger("relay.simulator")


class ModelSimulator(ModelInterface):
    """Configurable local model simulator capable of mimicking latency distributions,

    concurrency degradation, transient failures, and streaming tokens.

    Key Differences from Real Hosted Groq:
    1. Deterministic generation vs black-box autoregressive sampling.
    2. Synthetic token emission rates rather than LPUs (Language Processing Units).
    3. Perfect local state isolation vs multitenant cloud traffic contention.
    """

    def __init__(
        self,
        base_latency_ms: float = 280.0,
        latency_jitter_ms: float = 60.0,
        failure_probability: float = 0.0,
        timeout_probability: float = 0.0,
        load_slowdown_factor: float = 0.05,
        streaming_chunk_delay_ms: float = 12.0,
        random_seed: Optional[int] = 42,
        model_name: str = "simulated-relay-llm",
    ):
        self.base_latency_ms = base_latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self.failure_probability = failure_probability
        self.timeout_probability = timeout_probability
        self.load_slowdown_factor = load_slowdown_factor
        self.streaming_chunk_delay_ms = streaming_chunk_delay_ms
        self.model_name = model_name
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)

        self._in_flight = 0
        self._lock = asyncio.Lock()

    def set_seed(self, seed: int):
        self.random_seed = seed
        random.seed(seed)

    def _generate_synthetic_answer(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, bool]:
        """Generates domain-accurate answers or refusals based on prompt keywords and context."""
        prompt_lower = prompt.lower()

        # Out-of-scope refusal triggers
        if any(unsupported in prompt_lower for unsupported in [
            "apple vision", "vision pro", "gpt-4", "gpt 4", "battery life", "tesla", "spacex", "iphone"
        ]):
            refusal_text = (
                "REFUSAL: The research library contains papers on the Transformer architecture (1706.03762v7), "
                "Moshi real-time dialogue (2410.00037v1), and Dynamic Chunking (2507.07955v2). "
                "It does not contain information to answer this question."
            )
            return refusal_text, True

        # Attention Is All You Need domain
        if "d_model" in prompt_lower or ("base transformer" in prompt_lower and "dimension" in prompt_lower):
            ans = (
                "In the base Transformer architecture described in 'Attention Is All You Need' (arXiv:1706.03762v7, p. 3), "
                "the dimensionality of input and output embeddings is d_model = 512. "
                "All sub-layers in both the encoder and decoder produce outputs of dimension d_model = 512."
            )
            return ans, False

        if "encoder and decoder" in prompt_lower and ("layer" in prompt_lower or "how many" in prompt_lower):
            ans = (
                "According to 'Attention Is All You Need' (arXiv:1706.03762v7, p. 3), the base Transformer uses a stack "
                "of N = 6 identical layers in the encoder and N = 6 identical layers in the decoder."
            )
            return ans, False

        if "scaled dot-product" in prompt_lower or "square root of d_k" in prompt_lower:
            ans = (
                "As explained in 'Attention Is All You Need' (arXiv:1706.03762v7, p. 4), dividing by sqrt(d_k) prevents "
                "the dot products from growing excessively large for large values of d_k, which otherwise pushes the "
                "softmax function into regions with vanishingly small gradients."
            )
            return ans, False

        if "rope" in prompt_lower or "swiglu" in prompt_lower:
            ans = (
                "Neither RoPE nor SwiGLU was used. 'Attention Is All You Need' (arXiv:1706.03762v7, p. 5) employed "
                "fixed sinusoidal positional encodings (along with learned positional embeddings in experiments) "
                "and standard ReLU activations in the feed-forward sublayers."
            )
            return ans, False

        # Moshi domain
        if "mimi" in prompt_lower or ("sampling rate" in prompt_lower and "moshi" in prompt_lower):
            ans = (
                "In 'Moshi: A Speech-Text Foundation Model for Real-Time Dialogue' (arXiv:2410.00037v1, p. 4), "
                "the Mimi neural audio codec operates on 24 kHz audio and compresses it to a 12.5 Hz framerate "
                "using 8 residual vector quantizers (RVQ)."
            )
            return ans, False

        if "conversational dialogue" in prompt_lower or "multi-stream" in prompt_lower:
            ans = (
                "Moshi (arXiv:2410.00037v1, p. 2) models dialogue as a multi-stream audio language model operating on "
                "interleaved discrete speech tokens from both user and system audio streams simultaneously, "
                "eliminating conventional turn-taking handoffs."
            )
            return ans, False

        if "latency" in prompt_lower and "moshi" in prompt_lower:
            ans = (
                "Moshi (arXiv:2410.00037v1, p. 7) achieves a theoretical latency of 160ms (an 80ms frame stride plus 80ms lookahead) "
                "and an overall practical end-to-end latency of approximately 200ms on a single GPU."
            )
            return ans, False

        if "video" in prompt_lower and "moshi" in prompt_lower:
            return (
                "REFUSAL: The paper 'Moshi' (arXiv:2410.00037v1) is exclusively a speech-text foundation model. "
                "It does not utilize video footage or incorporate visual alignment.",
                True,
            )

        # Dynamic Chunking domain
        if "dynamic chunking" in prompt_lower and ("motivation" in prompt_lower or "fixed-size" in prompt_lower):
            ans = (
                "In 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling' (arXiv:2507.07955v2, p. 2), "
                "dynamic chunking adaptively selects chunk boundaries based on semantic density rather than rigid fixed windows, "
                "preventing unnatural token boundary splits."
            )
            return ans, False

        if "gradient" in prompt_lower and ("chunking" in prompt_lower or "collapse" in prompt_lower):
            ans = (
                "Dynamic Chunking (arXiv:2507.07955v2, p. 5) utilizes continuous relaxation and straight-through estimators "
                "to allow gradients to propagate through the discrete boundary selection decisions end-to-end."
            )
            return ans, False

        if "complexity" in prompt_lower or "compare" in prompt_lower:
            ans = (
                "Standard self-attention in the Transformer (arXiv:1706.03762v7) requires O(N^2) pairwise quadratic computation. "
                "In contrast, Dynamic Chunking (arXiv:2507.07955v2) reduces sequence complexity by hierarchically computing "
                "intra-chunk and inter-chunk attention over dynamically grouped token spans."
            )
            return ans, False

        # Generic grounded synthesis using prompt text
        ans = (
            f"Based on the retrieved research library: The technical analysis indicates that the query relates to "
            f"transformer architectures and sequence modeling techniques detailed in the retrieved papers."
        )
        return ans, False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> ModelResponse:
        t_start = time.perf_counter()

        async with self._lock:
            self._in_flight += 1
            current_flight = self._in_flight

        try:
            # 1. Simulate transient failure
            if self.failure_probability > 0 and random.random() < self.failure_probability:
                raise ConnectionError("Simulated upstream 503 Service Unavailable / Connection Drop")

            # 2. Simulate timeout
            if self.timeout_probability > 0 and random.random() < self.timeout_probability:
                await asyncio.sleep(2.5)
                raise TimeoutError("Simulated upstream gateway timeout")

            # 3. Compute load-dependent latency
            slowdown = 1.0 + (current_flight * self.load_slowdown_factor)
            jitter = random.uniform(-self.latency_jitter_ms, self.latency_jitter_ms)
            sim_latency_ms = max(50.0, (self.base_latency_ms + jitter) * slowdown)
            sim_latency_sec = sim_latency_ms / 1000.0

            # First token arrives after ~35% of total time
            first_token_sec = sim_latency_sec * 0.35
            await asyncio.sleep(first_token_sec)
            t_first = time.perf_counter()

            # Remaining generation time
            await asyncio.sleep(max(0.01, sim_latency_sec - first_token_sec))
            t_end = time.perf_counter()

            content, is_refusal = self._generate_synthetic_answer(prompt, system_prompt)
            tokens_in = int(len(prompt.split()) / 0.75) + 20
            tokens_out = int(len(content.split()) / 0.75)

            return ModelResponse(
                content=content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=(t_first - t_start) * 1000.0,
                total_ms=(t_end - t_start) * 1000.0,
                source="stand-in",
                model_name=self.model_name,
                refusal=is_refusal,
            )
        finally:
            async with self._lock:
                self._in_flight = max(0, self._in_flight - 1)

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> AsyncIterator[Dict[str, Any]]:
        t_start = time.perf_counter()
        content, _ = self._generate_synthetic_answer(prompt, system_prompt)
        words = content.split()

        # Emit initial latency delay
        base_delay_sec = max(0.05, self.base_latency_ms / 1000.0 * 0.35)
        await asyncio.sleep(base_delay_sec)

        is_first = True
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            t_now = time.perf_counter()
            first_ms = (t_now - t_start) * 1000.0 if is_first else None
            is_first = False

            yield {
                "delta": chunk,
                "first_token_ms": first_ms,
                "elapsed_ms": (t_now - t_start) * 1000.0,
            }
            await asyncio.sleep(self.streaming_chunk_delay_ms / 1000.0)
