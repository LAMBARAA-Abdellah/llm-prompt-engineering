"""
IMDB Movie Reviews dataset loader.

Loads, cleans, and prepares the IMDB sentiment dataset for evaluation.
Falls back to a built-in sample if download is unavailable.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import get_settings
from datasets.downloader import DatasetDownloader
from utils.logger import get_logger

logger = get_logger(__name__)


class IMDBLoader:
    """Load and preprocess IMDB sentiment dataset."""

    def __init__(self) -> None:
        settings = get_settings()
        self.datasets_dir = settings.datasets_dir
        self.output_dir = settings.output_dir

    def load(self, sample_size: Optional[int] = None, use_sample: bool = True) -> pd.DataFrame:
        """
        Load IMDB dataset into a DataFrame.

        Args:
            sample_size: Number of samples to return (None = all).
            use_sample: If True, use built-in sample when download fails.

        Returns:
            DataFrame with columns: text, label (positive/negative).
        """
        df = self._try_load_from_archive()
        if df is None and use_sample:
            logger.warning("Using built-in sample dataset")
            df = self._get_sample_data()

        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=get_settings().random_seed)

        return df.reset_index(drop=True)

    def _try_load_from_archive(self) -> Optional[pd.DataFrame]:
        """Attempt to load from downloaded tar.gz archive."""
        archive = self.datasets_dir / "aclImdb_v1.tar.gz"
        if not archive.exists():
            try:
                DatasetDownloader().download_imdb()
            except Exception as exc:
                logger.warning("IMDB download failed: %s", exc)
                return None

        if not archive.exists():
            return None

        records = []
        extract_dir = self.datasets_dir / "aclImdb"
        if not extract_dir.exists():
            logger.info("Extracting IMDB archive...")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(self.datasets_dir)

        for sentiment in ("pos", "neg"):
            folder = extract_dir / "train" / sentiment
            if not folder.exists():
                continue
            label = "positive" if sentiment == "pos" else "negative"
            for fpath in folder.glob("*.txt"):
                text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
                records.append({"text": text, "label": label})

        if not records:
            return None

        logger.info("Loaded %d IMDB reviews", len(records))
        return pd.DataFrame(records)

    @staticmethod
    def _get_sample_data() -> pd.DataFrame:
        """Built-in sample for offline/demo use."""
        return pd.DataFrame({
            "text": [
                "This movie was absolutely fantastic! Best film of the year.",
                "Terrible acting and a boring plot. Waste of time.",
                "A masterpiece of cinematography and storytelling.",
                "I fell asleep halfway through. Completely dull.",
                "The performances were outstanding and moving.",
                "Predictable and uninspired. Would not recommend.",
                "An emotional rollercoaster — loved every minute.",
                "Poor dialogue and weak character development.",
                "Visually stunning with a compelling narrative.",
                "One of the worst movies I've ever seen.",
                "Brilliant direction and a thought-provoking script.",
                "Mediocre at best. Nothing memorable.",
                "The soundtrack alone makes this worth watching.",
                "Confusing plot with no resolution. Very disappointing.",
                "A heartwarming story that stays with you.",
                "Overrated and overhyped. Expected much more.",
                "Perfect blend of action and emotion.",
                "The pacing was too slow for my taste.",
                "Incredible performances from the entire cast.",
                "Cheap production values and bad special effects.",
            ],
            "label": [
                "positive", "negative", "positive", "negative", "positive",
                "negative", "positive", "negative", "positive", "negative",
                "positive", "negative", "positive", "negative", "positive",
                "negative", "positive", "negative", "positive", "negative",
            ],
        })

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Basic text cleaning."""
        df = df.copy()
        df["text"] = df["text"].str.strip()
        df = df[df["text"].str.len() > 10]
        df = df.drop_duplicates(subset=["text"])
        return df.reset_index(drop=True)

    @staticmethod
    def statistics(df: pd.DataFrame) -> dict:
        """Compute dataset statistics."""
        return {
            "total_samples": len(df),
            "positive_count": (df["label"] == "positive").sum(),
            "negative_count": (df["label"] == "negative").sum(),
            "avg_text_length": df["text"].str.len().mean(),
            "min_text_length": df["text"].str.len().min(),
            "max_text_length": df["text"].str.len().max(),
        }

    def visualize(self, df: pd.DataFrame, save: bool = True) -> None:
        """Generate EDA visualizations."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Label distribution
        df["label"].value_counts().plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"])
        axes[0].set_title("Sentiment Distribution")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel("Count")

        # Text length distribution
        df["text_length"] = df["text"].str.len()
        for label, color in [("positive", "#2ecc71"), ("negative", "#e74c3c")]:
            subset = df[df["label"] == label]["text_length"]
            axes[1].hist(subset, bins=20, alpha=0.6, label=label, color=color)
        axes[1].set_title("Text Length Distribution")
        axes[1].set_xlabel("Character Count")
        axes[1].legend()

        # Box plot
        sns.boxplot(data=df, x="label", y="text_length", ax=axes[2],
                     palette={"positive": "#2ecc71", "negative": "#e74c3c"})
        axes[2].set_title("Text Length by Sentiment")

        plt.tight_layout()
        if save:
            path = self.output_dir / "imdb_eda.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved EDA plot to %s", path)
        plt.close()
