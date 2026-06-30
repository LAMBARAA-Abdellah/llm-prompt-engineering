"""
Model and provider configuration definitions.

Centralizes supported models for each LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    name: str
    provider: Provider
    max_tokens: int = 1024
    temperature: float = 0.7
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    description: str = ""


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider with its supported models."""

    provider: Provider
    models: List[ModelConfig] = field(default_factory=list)
    default_model: str = ""

    def get_model(self, name: str | None = None) -> ModelConfig:
        """Return model config by name or default."""
        target = name or self.default_model
        for model in self.models:
            if model.name == target:
                return model
        raise ValueError(f"Model '{target}' not found for provider {self.provider.value}")


# ---------------------------------------------------------------------------
# Pre-defined provider configurations
# ---------------------------------------------------------------------------

OPENAI_MODELS = ProviderConfig(
    provider=Provider.OPENAI,
    default_model="gpt-4o-mini",
    models=[
        ModelConfig(
            name="gpt-4o-mini",
            provider=Provider.OPENAI,
            max_tokens=4096,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            description="Fast, cost-effective GPT-4 variant",
        ),
        ModelConfig(
            name="gpt-4o",
            provider=Provider.OPENAI,
            max_tokens=4096,
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
            description="Most capable OpenAI model",
        ),
        ModelConfig(
            name="dall-e-3",
            provider=Provider.OPENAI,
            max_tokens=0,
            cost_per_1k_input=0.04,
            cost_per_1k_output=0.0,
            description="Text-to-image generation model",
        ),
    ],
)

OLLAMA_MODELS = ProviderConfig(
    provider=Provider.OLLAMA,
    default_model="llama3",
    models=[
        ModelConfig(
            name="llama3",
            provider=Provider.OLLAMA,
            max_tokens=4096,
            description="Meta Llama 3 - general purpose",
        ),
        ModelConfig(
            name="mistral",
            provider=Provider.OLLAMA,
            max_tokens=4096,
            description="Mistral 7B - efficient open model",
        ),
        ModelConfig(
            name="gemma",
            provider=Provider.OLLAMA,
            max_tokens=4096,
            description="Google Gemma - lightweight model",
        ),
    ],
)

GROQ_MODELS = ProviderConfig(
    provider=Provider.GROQ,
    default_model="llama-3.3-70b-versatile",
    models=[
        ModelConfig(
            name="llama-3.3-70b-versatile",
            provider=Provider.GROQ,
            max_tokens=8192,
            cost_per_1k_input=0.00059,
            cost_per_1k_output=0.00079,
            description="Llama 3.3 70B on Groq hardware",
        ),
        ModelConfig(
            name="mixtral-8x7b-32768",
            provider=Provider.GROQ,
            max_tokens=32768,
            cost_per_1k_input=0.00024,
            cost_per_1k_output=0.00024,
            description="Mixtral 8x7B MoE model",
        ),
    ],
)

PROVIDER_REGISTRY: Dict[Provider, ProviderConfig] = {
    Provider.OPENAI: OPENAI_MODELS,
    Provider.OLLAMA: OLLAMA_MODELS,
    Provider.GROQ: GROQ_MODELS,
}


def get_provider_config(provider: Provider | str) -> ProviderConfig:
    """Retrieve provider configuration by enum or string name."""
    if isinstance(provider, str):
        provider = Provider(provider.lower())
    return PROVIDER_REGISTRY[provider]
