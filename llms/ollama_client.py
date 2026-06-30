"""
Ollama LLM client for local model inference.

Supports:
- llama3
- mistral
- gemma

Requires Ollama server running locally (default: http://localhost:11434).
"""

from __future__ import annotations

from typing import Any, Dict, List

import ollama

from config.models import OLLAMA_MODELS, Provider
from config.settings import get_settings
from llms.base_client import BaseLLMClient, LLMResponse
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_MODELS = ["llama3", "mistral", "gemma"]


class OllamaClient(BaseLLMClient):
    """Client for local Ollama LLM inference."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        model_name = model or settings.ollama_model

        if model_name not in SUPPORTED_MODELS:
            logger.warning(
                "Model '%s' not in supported list %s. Attempting anyway.",
                model_name,
                SUPPORTED_MODELS,
            )

        super().__init__(
            model=model_name,
            max_retries=settings.max_retries,
            retry_delay=settings.retry_delay,
            temperature=kwargs.get("temperature", settings.default_temperature),
            max_tokens=kwargs.get("max_tokens", settings.default_max_tokens),
        )
        self.provider_name = "ollama"
        self.base_url = base_url or settings.ollama_base_url
        self._client = ollama.Client(host=self.base_url)
        self._model_config = OLLAMA_MODELS.get_model(self.model)
        logger.info("Ollama client initialized | model=%s | host=%s", self.model, self.base_url)

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute Ollama chat completion."""
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        )

        content = response.get("message", {}).get("content", "")
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
            raw_response=response,
            metadata={"cost_estimate": 0.0},  # Local inference is free
        )

    def list_models(self) -> List[str]:
        """List models available on the local Ollama server."""
        try:
            models = self._client.list()
            return [m.get("name", "") for m in models.get("models", [])]
        except Exception as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            return []

    def pull_model(self, model_name: str) -> None:
        """Pull/download a model to the local Ollama server."""
        logger.info("Pulling Ollama model: %s", model_name)
        self._client.pull(model_name)
        logger.info("Model %s pulled successfully", model_name)
