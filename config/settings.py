"""
Application settings loaded from environment variables.

Uses python-dotenv to read configuration from .env file.
Never hardcodes API keys or secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    """Central configuration for the entire application."""

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_image_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
    )

    # Groq
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # Ollama
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))

    # Application
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("RETRY_DELAY", "2")))
    default_temperature: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    )
    default_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
    )

    # Evaluation
    eval_sample_size: int = field(
        default_factory=lambda: int(os.getenv("EVAL_SAMPLE_SIZE", "100"))
    )
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))

    # Paths
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("OUTPUT_DIR", "outputs")
    )
    images_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("IMAGES_DIR", "images")
    )
    datasets_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATASETS_DIR", "datasets/raw")
    )

    def __post_init__(self) -> None:
        """Ensure output directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def validate_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")

    def validate_groq(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.groq_api_key and self.groq_api_key != "your_groq_api_key_here")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
