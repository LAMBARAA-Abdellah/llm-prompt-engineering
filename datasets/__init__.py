"""Dataset loading and management package."""

from datasets.downloader import DatasetDownloader
from datasets.imdb_loader import IMDBLoader
from datasets.spam_loader import SpamLoader

__all__ = ["DatasetDownloader", "IMDBLoader", "SpamLoader"]
