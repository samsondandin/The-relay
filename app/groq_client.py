"""Groq API client adapter implementing ModelInterface with latency and token tracking."""

import os
import time
import logging
from typing import Optional, Dict, Any, AsyncIterator
import httpx
from groq import AsyncGroq, RateLimitError, APIConnectionError, APIStatusError, APITimeoutError

from app.model_interface import ModelInterface, ModelResponse
from app.rate_limiter import RateLimitExceededException, DailyBudgetExhaustedException

logger = logging.getLogger("relay.groq_client")


class GroqClient(ModelInterface):
    """Adapter for hosted Groq models with telemetry, streaming, and error mapping."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        timeout_seconds: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

        if self.api_key:
            self.client = AsyncGroq(api_key=self.api_key, timeout=timeout_seconds)
        else:
            self.client = None

    def _ensure_client(self):
        if not self.client:
            raise ValueError(
                "GROQ_API_KEY is not configured. Set the GROQ_API_KEY environment variable or use the simulator."
            )

    def _parse_headers(self, response_headers) -> Dict[str, str]:
        if not response_headers:
            return {}
        relevant_keys = [
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
            "retry-after",
        ]
        return {k: str(response_headers.get(k)) for k in relevant_keys if k in response_headers}

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> ModelResponse:
        self._ensure_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t_start = time.perf_counter()
        t_first_token: Optional[float] = None

        try:
            # We use stream=True even for full generation so we accurately capture first-token TTFT
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            collected_content = []
            async for chunk in stream:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                delta = chunk.choices[0].delta.content or ""
                collected_content.append(delta)

            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            first_token_ms = (t_first_token - t_start) * 1000.0 if t_first_token else None

            full_text = "".join(collected_content)
            # Estimate tokens if provider stream does not include raw usage
            tokens_in = int(len(prompt.split()) / 0.75) + 30
            tokens_out = int(len(full_text.split()) / 0.75)

            return ModelResponse(
                content=full_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                source="real",
                model_name=self.model_name,
                refusal="REFUSAL:" in full_text.upper() or "I CANNOT ANSWER" in full_text.upper(),
            )

        except RateLimitError as e:
            headers = getattr(e, "response", None)
            retry_after = 2.0
            if headers and hasattr(headers, "headers"):
                retry_header = headers.headers.get("retry-after")
                if retry_header:
                    try:
                        retry_after = float(retry_header)
                    except ValueError:
                        pass
            logger.warning(f"Groq RateLimitError 429: {e}")
            raise RateLimitExceededException(f"Groq API rate limit hit: {e}", retry_after=retry_after)

        except (APITimeoutError, httpx.TimeoutException) as e:
            logger.warning(f"Groq APITimeoutError: {e}")
            raise TimeoutError(f"Groq request timed out: {e}")

        except APIConnectionError as e:
            logger.warning(f"Groq APIConnectionError: {e}")
            raise ConnectionError(f"Groq connection error: {e}")

        except APIStatusError as e:
            logger.warning(f"Groq APIStatusError {e.status_code}: {e.message}")
            if e.status_code == 429:
                raise RateLimitExceededException(f"Groq 429: {e.message}", retry_after=3.0)
            raise e

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> AsyncIterator[Dict[str, Any]]:
        self._ensure_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t_start = time.perf_counter()
        is_first = True

        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            t_now = time.perf_counter()
            first_token_ms = (t_now - t_start) * 1000.0 if is_first else None
            is_first = False

            yield {
                "delta": delta,
                "first_token_ms": first_token_ms,
                "elapsed_ms": (t_now - t_start) * 1000.0,
            }

