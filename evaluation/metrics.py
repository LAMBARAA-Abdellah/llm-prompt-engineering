"""
Evaluation metrics for prompt engineering experiments.

Computes: Accuracy, Precision, Recall, F1 Score, Confusion Matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, precision, recall, f1.
    """
    # Normalize labels
    y_true_norm = [l.strip().lower() for l in y_true]
    y_pred_norm = [_normalize_prediction(p) for p in y_pred]

    metrics = {
        "accuracy": accuracy_score(y_true_norm, y_pred_norm),
        "precision": precision_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0),
        "recall": recall_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0),
    }
    logger.info("Metrics: %s", metrics)
    return metrics


def _normalize_prediction(pred: str) -> str:
    """Extract label from LLM response text."""
    pred = pred.strip().lower()
    for label in ("positive", "negative", "ham", "spam"):
        if label in pred:
            return label
    return pred.split()[0] if pred else "unknown"


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Plot and optionally save confusion matrix."""
    y_true_norm = [l.strip().lower() for l in y_true]
    y_pred_norm = [_normalize_prediction(p) for p in y_pred]

    if labels is None:
        labels = sorted(set(y_true_norm + y_pred_norm))

    cm = confusion_matrix(y_true_norm, y_pred_norm, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    path = save_path or str(get_settings().output_dir / "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Confusion matrix saved to %s", path)
    plt.close()


def classification_report_str(y_true: List[str], y_pred: List[str]) -> str:
    """Return sklearn classification report as string."""
    y_true_norm = [l.strip().lower() for l in y_true]
    y_pred_norm = [_normalize_prediction(p) for p in y_pred]
    return classification_report(y_true_norm, y_pred_norm, zero_division=0)
