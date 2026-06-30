"""
Dataset downloader utility.

Downloads public NLP datasets from reliable sources.
"""

from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetDownloader:
    """Download and cache public datasets."""

    DATASET_URLS = {
        "imdb": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        "sms_spam": "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    }

    def __init__(self) -> None:
        settings = get_settings()
        self.datasets_dir = settings.datasets_dir
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, filename: str) -> Path:
        """
        Download a file with progress bar.

        Args:
            url: Source URL.
            filename: Local filename to save as.

        Returns:
            Path to downloaded file.
        """
        dest = self.datasets_dir / filename
        if dest.exists():
            logger.info("File already exists: %s", dest)
            return dest

        logger.info("Downloading %s...", url)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info("Downloaded to %s", dest)
        return dest

    def download_imdb(self) -> Path:
        """Download IMDB sentiment dataset."""
        return self.download_file(self.DATASET_URLS["imdb"], "aclImdb_v1.tar.gz")

    def download_sms_spam(self) -> Path:
        """Download SMS Spam Collection dataset."""
        return self.download_file(self.DATASET_URLS["sms_spam"], "smsspamcollection.zip")
