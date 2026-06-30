"""
Groq LLM client for fast cloud inference.

Supports:
- llama-3.3-70b-versatile
- mixtral-8x7b-32768
"""

from __future__ import annotations

from typing import Any, Dict, List

from groq import Groq, APIError, RateLimitError, AuthenticationError

from config.models import GROQ_MODELS
from config.settings import get_settings
from llms.base_client import BaseLLMClient, LLMResponse
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_MODELS = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]


class GroqClient(BaseLLMClient):
    """Groq API client for fast LLM inference."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        super().__init__(
            model=model or settings.groq_model,
            max_retries=settings.max_retries,
            retry_delay=settings.retry_delay,
            temperature=kwargs.get("temperature", settings.default_temperature),
            max_tokens=kwargs.get("max_tokens", settings.default_max_tokens),
        )
        self.provider_name = "groq"
        key = api_key or settings.groq_api_key

        if not key or key == "your_groq_api_key_here":
            raise AuthenticationError(
                message="Groq API key not configured. Set GROQ_API_KEY in .env"
            )

        self.client = Groq(api_key=key)
        self._model_config = GROQ_MODELS.get_model(self.model)
        logger.info("Groq client initialized with model: %s", self.model)

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute Groq chat completion."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            choice = response.choices[0]
            usage = response.usage

            llm_response = LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                provider=self.provider_name,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                raw_response=response,
            )
            llm_response.metadata["cost_estimate"] = self._estimate_cost(llm_response)
            return llm_response

        except RateLimitError:
            logger.error("Groq rate limit exceeded")
            raise
        except APIError as exc:
            logger.error("Groq API error: %s", exc)
            raise

    def _estimate_cost(self, response: LLMResponse) -> float:
        """Calculate estimated cost based on Groq pricing."""
        cfg = self._model_config
        input_cost = (response.prompt_tokens / 1000) * cfg.cost_per_1k_input
        output_cost = (response.completion_tokens / 1000) * cfg.cost_per_1k_output
        return round(input_cost + output_cost, 6)
