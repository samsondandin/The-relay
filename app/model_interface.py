"""Abstract Model Interface defining the contract for both Groq and Simulator backends."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncIterator, Literal
from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    content: str = Field(..., description="Generated text content")
    tokens_in: int = Field(default=0, description="Prompt tokens consumed")
    tokens_out: int = Field(default=0, description="Completion tokens generated")
    first_token_ms: Optional[float] = Field(default=None, description="Time to first token in milliseconds")
    total_ms: float = Field(..., description="Total execution latency in milliseconds")
    source: Literal["real", "stand-in"] = Field(..., description="Model backend origin")
    model_name: str = Field(default="", description="Model identifier used")
    raw_headers: Dict[str, str] = Field(default_factory=dict, description="Response headers from provider")
    refusal: bool = Field(default=False, description="Whether the response is a refusal")


class ModelInterface(ABC):
    """Abstract base class for LLM model adapters."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> ModelResponse:
        """Execute a complete generation request."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream generation tokens as they arrive."""
        pass

