"""
Prompt evaluation orchestrator.

Runs multiple prompting techniques against a dataset and collects results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from evaluation.metrics import compute_metrics, plot_confusion_matrix, _normalize_prediction
from llms.base_client import BaseLLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TechniqueResult:
    """Results from evaluating one prompting technique."""

    technique: str
    predictions: List[str] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def avg_time(self) -> float:
        return sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)

    @property
    def total_cost(self) -> float:
        return sum(self.costs)


class PromptEvaluator:
    """Evaluate prompting techniques on a labeled dataset."""

    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client
        self.results: Dict[str, TechniqueResult] = {}

    def evaluate_technique(
        self,
        technique_name: str,
        df: pd.DataFrame,
        predict_fn: Callable[[BaseLLMClient, str], Any],
        text_col: str = "text",
        label_col: str = "label",
        show_progress: bool = True,
    ) -> TechniqueResult:
        """
        Evaluate a single prompting technique.

        Args:
            technique_name: Name identifier for the technique.
            df: DataFrame with text and label columns.
            predict_fn: Function(client, text) -> LLMResponse.
            text_col: Column name for input text.
            label_col: Column name for ground truth labels.
            show_progress: Show tqdm progress bar.

        Returns:
            TechniqueResult with predictions and metrics.
        """
        result = TechniqueResult(technique=technique_name)
        y_true = df[label_col].tolist()

        iterator = df[text_col].tolist()
        if show_progress:
            iterator = tqdm(iterator, desc=f"Evaluating {technique_name}")

        for text in iterator:
            try:
                response = predict_fn(self.client, text)
                pred = _normalize_prediction(response.content)
                result.predictions.append(pred)
                result.execution_times.append(response.execution_time)
                result.token_counts.append(response.total_tokens)
                result.costs.append(response.metadata.get("cost_estimate", 0.0))
            except Exception as exc:
                logger.error("Prediction failed for '%s...': %s", text[:30], exc)
                result.predictions.append("unknown")
                result.execution_times.append(0.0)
                result.token_counts.append(0)
                result.costs.append(0.0)

        result.metrics = compute_metrics(y_true, result.predictions)
        self.results[technique_name] = result
        logger.info(
            "%s | Accuracy: %.4f | Avg Time: %.2fs | Total Cost: $%.4f",
            technique_name,
            result.metrics["accuracy"],
            result.avg_time,
            result.total_cost,
        )
        return result

    def compare_all(self) -> pd.DataFrame:
        """Generate comparison DataFrame across all evaluated techniques."""
        rows = []
        for name, result in self.results.items():
            rows.append({
                "Technique": name,
                "Accuracy": result.metrics.get("accuracy", 0),
                "Precision": result.metrics.get("precision", 0),
                "Recall": result.metrics.get("recall", 0),
                "F1 Score": result.metrics.get("f1_score", 0),
                "Avg Time (s)": round(result.avg_time, 3),
                "Total Tokens": result.total_tokens,
                "Total Cost ($)": round(result.total_cost, 4),
            })
        return pd.DataFrame(rows)

    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """Generate comparison charts for all techniques."""
        import matplotlib.pyplot as plt

        df = self.compare_all()
        if df.empty:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Accuracy comparison
        df.plot(x="Technique", y="Accuracy", kind="bar", ax=axes[0], color="#3498db", legend=False)
        axes[0].set_title("Accuracy Comparison")
        axes[0].set_ylim(0, 1.1)
        axes[0].tick_params(axis="x", rotation=45)

        # Speed comparison
        df.plot(x="Technique", y="Avg Time (s)", kind="bar", ax=axes[1], color="#e67e22", legend=False)
        axes[1].set_title("Average Execution Time")
        axes[1].tick_params(axis="x", rotation=45)

        # Cost comparison
        df.plot(x="Technique", y="Total Cost ($)", kind="bar", ax=axes[2], color="#2ecc71", legend=False)
        axes[2].set_title("Total Cost Estimate")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        from config.settings import get_settings
        path = save_path or str(get_settings().output_dir / "technique_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Comparison chart saved to %s", path)
        plt.close()

    def plot_confusion_for_technique(
        self,
        technique_name: str,
        y_true: List[str],
        labels: List[str] | None = None,
    ) -> None:
        """Plot confusion matrix for a specific technique."""
        if technique_name not in self.results:
            return
        result = self.results[technique_name]
        from config.settings import get_settings
        path = str(get_settings().output_dir / f"cm_{technique_name}.png")
        plot_confusion_matrix(y_true, result.predictions, labels=labels,
                              title=f"Confusion Matrix — {technique_name}", save_path=path)
