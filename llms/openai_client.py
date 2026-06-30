"""
OpenAI LLM client implementation.

Features:
- Authentication via API key from environment
- Chat completion (GPT models)
- Image generation (DALL-E)
- Error handling with retry (inherited)
- Execution time tracking
- Token usage reporting
- Cost estimation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, RateLimitError, AuthenticationError

from config.models import OPENAI_MODELS, Provider
from config.settings import get_settings
from llms.base_client import BaseLLMClient, LLMResponse
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client for chat and image generation."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        super().__init__(
            model=model or settings.openai_model,
            max_retries=settings.max_retries,
            retry_delay=settings.retry_delay,
            temperature=kwargs.get("temperature", settings.default_temperature),
            max_tokens=kwargs.get("max_tokens", settings.default_max_tokens),
        )
        self.provider_name = "openai"
        key = api_key or settings.openai_api_key

        if not key or key == "your_openai_api_key_here":
            raise AuthenticationError(
                message="OpenAI API key not configured. Set OPENAI_API_KEY in .env",
                response=None,
                body=None,
            )

        self.client = OpenAI(api_key=key)
        self._model_config = OPENAI_MODELS.get_model(self.model)
        logger.info("OpenAI client initialized with model: %s", self.model)

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute OpenAI chat completion."""
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
            logger.error("OpenAI rate limit exceeded")
            raise
        except APIError as exc:
            logger.error("OpenAI API error: %s", exc.message)
            raise

    def _estimate_cost(self, response: LLMResponse) -> float:
        """Calculate estimated cost based on token usage."""
        cfg = self._model_config
        input_cost = (response.prompt_tokens / 1000) * cfg.cost_per_1k_input
        output_cost = (response.completion_tokens / 1000) * cfg.cost_per_1k_output
        return round(input_cost + output_cost, 6)

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> List[str]:
        """
        Generate images using DALL-E.

        Args:
            prompt: Text description of the desired image.
            size: Image dimensions (1024x1024, 1792x1024, 1024x1792).
            quality: 'standard' or 'hd'.
            n: Number of images to generate.

        Returns:
            List of image URLs.
        """
        settings = get_settings()
        image_model = settings.openai_image_model

        logger.info("Generating image with %s: %s...", image_model, prompt[:50])
        response = self.client.images.generate(
            model=image_model,
            prompt=prompt,
            size=size,  # type: ignore[arg-type]
            quality=quality,  # type: ignore[arg-type]
            n=n,
        )
        urls = [img.url for img in response.data if img.url]
        logger.info("Generated %d image(s)", len(urls))
        return urls

    def edit_image(
        self,
        image_path: str,
        prompt: str,
        mask_path: Optional[str] = None,
        size: str = "1024x1024",
    ) -> List[str]:
        """
        Edit an existing image using DALL-E (image-to-image).

        Args:
            image_path: Path to the source PNG image.
            prompt: Description of desired edits.
            mask_path: Optional mask PNG indicating edit regions.
            size: Output image size.

        Returns:
            List of edited image URLs.
        """
        with open(image_path, "rb") as img_file:
            kwargs: Dict[str, Any] = {
                "model": "dall-e-2",
                "image": img_file,
                "prompt": prompt,
                "n": 1,
                "size": size,
            }
            if mask_path:
                with open(mask_path, "rb") as mask_file:
                    kwargs["mask"] = mask_file
                    response = self.client.images.edit(**kwargs)
            else:
                response = self.client.images.edit(**kwargs)

        return [img.url for img in response.data if img.url]
