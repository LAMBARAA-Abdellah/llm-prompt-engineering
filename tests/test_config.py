"""Tests for configuration module."""

import os
from pathlib import Path

import pytest


def test_settings_load():
    """Settings should load without error."""
    from config.settings import Settings
    settings = Settings()
    assert settings.project_root.exists()
    assert settings.output_dir.exists()


def test_settings_paths_created():
    """Output directories should be created on init."""
    from config.settings import Settings
    settings = Settings()
    assert settings.images_dir.is_dir()
    assert settings.datasets_dir.is_dir()


def test_model_registry():
    """Provider registry should contain all three providers."""
    from config.models import Provider, PROVIDER_REGISTRY
    assert Provider.OPENAI in PROVIDER_REGISTRY
    assert Provider.OLLAMA in PROVIDER_REGISTRY
    assert Provider.GROQ in PROVIDER_REGISTRY


def test_openai_models():
    """OpenAI should have gpt-4o-mini as default."""
    from config.models import OPENAI_MODELS
    assert OPENAI_MODELS.default_model == "gpt-4o-mini"
    assert len(OPENAI_MODELS.models) >= 2


def test_ollama_supported_models():
    """Ollama should support llama3, mistral, gemma."""
    from config.models import OLLAMA_MODELS
    model_names = [m.name for m in OLLAMA_MODELS.models]
    assert "llama3" in model_names
    assert "mistral" in model_names
    assert "gemma" in model_names
