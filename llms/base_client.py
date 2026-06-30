"""
Abstract base class for all LLM clients.

Defines the common interface for chat completion, error handling,
retry logic, execution time tracking, and token usage reporting.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    provider: str
    execution_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost_estimate(self) -> float:
        """Estimated cost based on token usage (override in subclasses)."""
        return 0.0


class BaseLLMClient(ABC):
    """
    Abstract base for LLM provider clients.

    Implements retry logic and execution time measurement.
    Subclasses must implement _chat_completion.
    """

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_name = "base"

    @abstractmethod
    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a single chat completion request."""
        ...

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Chat completion with automatic retry on failure.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse with content, timing, and token usage.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.perf_counter()
                response = self._chat_completion(messages, **kwargs)
                response.execution_time = time.perf_counter() - start
                logger.info(
                    "%s | model=%s | time=%.2fs | tokens=%d",
                    self.provider_name,
                    self.model,
                    response.execution_time,
                    response.total_tokens,
                )
                return response
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt,
                    self.max_retries,
                    self.provider_name,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"{self.provider_name} failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def simple_prompt(self, prompt: str, system: str = "") -> LLMResponse:
        """Convenience method for single-turn prompts."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)
