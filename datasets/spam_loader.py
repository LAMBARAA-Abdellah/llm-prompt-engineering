"""
SMS Spam Collection dataset loader.

Loads and prepares the UCI SMS Spam dataset for spam detection evaluation.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from config.settings import get_settings
from datasets.downloader import DatasetDownloader
from utils.logger import get_logger

logger = get_logger(__name__)


class SpamLoader:
    """Load and preprocess SMS Spam dataset."""

    def __init__(self) -> None:
        settings = get_settings()
        self.datasets_dir = settings.datasets_dir
        self.output_dir = settings.output_dir

    def load(self, sample_size: Optional[int] = None, use_sample: bool = True) -> pd.DataFrame:
        """
        Load SMS spam dataset.

        Returns:
            DataFrame with columns: text, label (ham/spam).
        """
        df = self._try_load_from_archive()
        if df is None and use_sample:
            logger.warning("Using built-in spam sample dataset")
            df = self._get_sample_data()

        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=get_settings().random_seed)

        return df.reset_index(drop=True)

    def _try_load_from_archive(self) -> Optional[pd.DataFrame]:
        """Load from downloaded zip archive."""
        archive = self.datasets_dir / "smsspamcollection.zip"
        if not archive.exists():
            try:
                DatasetDownloader().download_sms_spam()
            except Exception as exc:
                logger.warning("SMS spam download failed: %s", exc)
                return None

        if not archive.exists():
            return None

        extract_dir = self.datasets_dir / "SMSSpamCollection"
        if not extract_dir.exists():
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(self.datasets_dir)

        data_file = self.datasets_dir / "SMSSpamCollection"
        if not data_file.exists():
            for f in self.datasets_dir.rglob("SMSSpamCollection"):
                if f.is_file():
                    data_file = f
                    break

        if not data_file.exists():
            return None

        records = []
        with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    label, text = parts
                    records.append({"label": label, "text": text})

        logger.info("Loaded %d SMS messages", len(records))
        return pd.DataFrame(records)

    @staticmethod
    def _get_sample_data() -> pd.DataFrame:
        """Built-in sample for offline/demo use."""
        return pd.DataFrame({
            "text": [
                "Hey, are we still meeting for lunch tomorrow?",
                "FREE entry to win a £1000 prize! Text WIN to 81234 now!",
                "Can you pick up milk on your way home?",
                "Congratulations! You've been selected for a free iPhone. Click here!",
                "Meeting rescheduled to 3pm. See you then.",
                "URGENT: Your bank account has been compromised. Verify now at fake-bank.com",
                "Thanks for dinner last night, it was lovely.",
                "Get 50% off all items! Limited time offer. Reply STOP to unsubscribe.",
                "What time is the movie tonight?",
                "You have won £5000! Call this number to claim your prize immediately!",
                "Don't forget to submit the assignment by Friday.",
                "Lowest rate guaranteed on home loans. Apply today!",
                "Happy birthday! Hope you have a wonderful day.",
                "Click here to claim your free gift voucher worth $500!!!",
                "The project presentation is on Monday at 10am.",
                "WINNER!! You have been chosen to receive a £900 prize reward!",
                "I'll be running 10 minutes late, sorry!",
                "Exclusive deal just for you! Buy 1 get 1 free. Text YES now.",
                "Can you send me the report when you get a chance?",
                "Your package is ready for collection at the post office.",
            ],
            "label": [
                "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
                "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
                "ham", "spam", "ham", "ham",
            ],
        })

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning."""
        df = df.copy()
        df["text"] = df["text"].str.strip()
        df = df[df["text"].str.len() > 3]
        df = df.drop_duplicates(subset=["text"])
        return df.reset_index(drop=True)

    @staticmethod
    def statistics(df: pd.DataFrame) -> dict:
        """Compute dataset statistics."""
        return {
            "total_samples": len(df),
            "ham_count": (df["label"] == "ham").sum(),
            "spam_count": (df["label"] == "spam").sum(),
            "avg_text_length": df["text"].str.len().mean(),
        }

    def visualize(self, df: pd.DataFrame, save: bool = True) -> None:
        """Generate EDA visualizations."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        df["label"].value_counts().plot(kind="bar", ax=axes[0], color=["#3498db", "#e74c3c"])
        axes[0].set_title("Ham vs Spam Distribution")
        axes[0].set_xlabel("Label")

        df["text_length"] = df["text"].str.len()
        for label, color in [("ham", "#3498db"), ("spam", "#e74c3c")]:
            subset = df[df["label"] == label]["text_length"]
            axes[1].hist(subset, bins=20, alpha=0.6, label=label, color=color)
        axes[1].set_title("Message Length Distribution")
        axes[1].legend()

        plt.tight_layout()
        if save:
            path = self.output_dir / "spam_eda.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved spam EDA to %s", path)
        plt.close()
