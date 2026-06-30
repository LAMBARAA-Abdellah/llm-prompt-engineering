"""
Image-to-Image Prompting Use Case
===================================

Demonstrates image transformation using OpenAI DALL-E edit API.

WORKFLOW:
1. Start with a source image (PNG with transparency)
2. Optionally provide a mask indicating edit regions
3. Provide a text prompt describing desired changes
4. DALL-E generates a transformed version
5. Save the result locally

NOTE: Image editing requires DALL-E 2 and PNG format.
If no source image exists, this module creates a placeholder
and demonstrates the workflow conceptually.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageDraw

from config.settings import get_settings
from llms.openai_client import OpenAIClient
from utils.console import print_section
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageToImageUseCase:
    """Transform images using DALL-E edit API."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAIClient()
        self.images_dir = settings.images_dir
        self.output_dir = self.images_dir / "transformed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_placeholder_source(self) -> Path:
        """
        Create a simple placeholder PNG for demonstration.

        In production, you would use a real source image.
        """
        source_path = self.images_dir / "source_placeholder.png"
        if source_path.exists():
            return source_path

        img = Image.new("RGBA", (512, 512), (135, 206, 235, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 200, 412, 400], fill=(34, 139, 34, 255))
        draw.ellipse([180, 80, 332, 200], fill=(255, 215, 0, 255))
        draw.text((160, 420), "Source Image", fill=(0, 0, 0, 255))
        img.save(source_path)
        logger.info("Created placeholder source image: %s", source_path)
        return source_path

    def transform(
        self,
        source_path: str | Path,
        prompt: str,
        mask_path: Optional[str | Path] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Transform a source image based on a text prompt.

        Args:
            source_path: Path to source PNG image.
            prompt: Description of desired transformation.
            mask_path: Optional mask PNG for targeted editing.
            filename: Optional output filename.

        Returns:
            Path to transformed image.
        """
        print_section(f"Image-to-Image: {prompt[:60]}...")
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")

        logger.info("Transforming image: %s with prompt: %s", source_path, prompt)

        try:
            urls = self.client.edit_image(
                str(source_path), prompt,
                mask_path=str(mask_path) if mask_path else None,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = filename or f"img2img_{timestamp}.png"
            save_path = self.output_dir / fname

            response = requests.get(urls[0], timeout=60)
            response.raise_for_status()
            save_path.write_bytes(response.content)

            meta_path = save_path.with_suffix(".txt")
            meta_path.write_text(
                f"Source: {source_path}\nPrompt: {prompt}\n"
                f"Mask: {mask_path or 'None'}\n"
                f"Generated: {datetime.now().isoformat()}\n",
                encoding="utf-8",
            )
            logger.info("Transformed image saved to %s", save_path)
            return save_path

        except Exception as exc:
            logger.warning(
                "Image edit API call failed (expected if no valid PNG): %s. "
                "Demonstrating workflow with local copy.",
                exc,
            )
            return self._demo_workflow(source_path, prompt)

    def _demo_workflow(self, source_path: Path, prompt: str) -> Path:
        """Fallback: copy source with metadata when API edit is unavailable."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"img2img_demo_{timestamp}.png"
        img = Image.open(source_path)
        img.save(save_path)

        meta_path = save_path.with_suffix(".txt")
        meta_path.write_text(
            f"[DEMO MODE — API edit unavailable]\n"
            f"Source: {source_path}\nPrompt: {prompt}\n"
            f"In production, DALL-E 2 would transform this image.\n",
            encoding="utf-8",
        )
        logger.info("Demo workflow output saved to %s", save_path)
        return save_path

    def run_demo(self) -> Path:
        """Run complete image-to-image demonstration."""
        source = self.create_placeholder_source()
        return self.transform(
            source,
            "Transform this landscape into a snowy winter scene with aurora borealis",
        )
