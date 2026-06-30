"""
Text-to-Image Prompting Use Case
===================================

Uses OpenAI DALL-E API to generate images from text prompts.
Images are automatically saved to the images/ directory.

WORKFLOW:
1. Craft a descriptive text prompt
2. Send to OpenAI Image API (DALL-E 3)
3. Download generated image
4. Save locally with metadata
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests

from config.settings import get_settings
from llms.openai_client import OpenAIClient
from utils.console import print_section
from utils.logger import get_logger

logger = get_logger(__name__)


class TextToImageUseCase:
    """Generate images from text prompts using DALL-E."""

    EXAMPLE_PROMPTS = [
        "A serene Japanese garden with cherry blossoms at sunset, watercolor style",
        "A futuristic city skyline with flying cars, cyberpunk aesthetic, neon lights",
        "An academic workspace with books, a laptop showing Python code, and coffee",
    ]

    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAIClient()
        self.images_dir = settings.images_dir / "generated"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate an image from a text prompt and save it locally.

        Args:
            prompt: Text description of desired image.
            size: Image dimensions.
            quality: 'standard' or 'hd'.
            filename: Optional custom filename.

        Returns:
            Path to saved image file.
        """
        print_section(f"Text-to-Image: {prompt[:60]}...")
        logger.info("Generating image for prompt: %s", prompt)

        urls = self.client.generate_image(prompt, size=size, quality=quality)
        if not urls:
            raise RuntimeError("No image URLs returned from API")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = filename or f"txt2img_{timestamp}.png"
        save_path = self.images_dir / fname

        response = requests.get(urls[0], timeout=60)
        response.raise_for_status()
        save_path.write_bytes(response.content)

        # Save prompt metadata
        meta_path = save_path.with_suffix(".txt")
        meta_path.write_text(
            f"Prompt: {prompt}\nSize: {size}\nQuality: {quality}\n"
            f"Generated: {datetime.now().isoformat()}\n",
            encoding="utf-8",
        )

        logger.info("Image saved to %s", save_path)
        return save_path

    def generate_examples(self) -> List[Path]:
        """Generate all example prompts."""
        paths = []
        for i, prompt in enumerate(self.EXAMPLE_PROMPTS):
            path = self.generate(prompt, filename=f"example_{i + 1}.png")
            paths.append(path)
        return paths
