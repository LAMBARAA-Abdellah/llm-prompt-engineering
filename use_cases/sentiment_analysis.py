"""
Sentiment Analysis Use Case
============================

Complete workflow:
1. Load IMDB dataset
2. Clean and visualize
3. Apply multiple prompting techniques
4. Evaluate against gold examples
5. Compare results
"""

from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from datasets.imdb_loader import IMDBLoader
from evaluation.evaluator import PromptEvaluator
from llms.base_client import BaseLLMClient, LLMResponse
from prompts.chain_of_thought_prompt import ChainOfThoughtPrompt, SENTIMENT_COT_TASK
from prompts.few_shot_prompt import FewShotPrompt, SENTIMENT_FEW_SHOT_EXAMPLES, SENTIMENT_TASK
from prompts.role_prompt import RolePrompt
from prompts.self_consistency_prompt import SelfConsistencyPrompt
from prompts.simple_prompt import SimplePrompt
from prompts.zero_shot_prompt import ZeroShotPrompt, SENTIMENT_ZERO_SHOT
from use_cases.gold_examples import gold_to_few_shot_examples
from utils.console import print_metrics_summary, print_results_table, print_section
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalysisUseCase:
    """End-to-end sentiment analysis with prompt engineering comparison."""

    TASK = "Classify the sentiment as 'positive' or 'negative'."

    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client
        self.loader = IMDBLoader()
        self.evaluator = PromptEvaluator(client)

    def load_and_prepare(self, sample_size: int = 20) -> pd.DataFrame:
        """Load, clean, and sample IMDB dataset."""
        print_section("Loading IMDB Sentiment Dataset")
        df = self.loader.load(sample_size=sample_size)
        df = self.loader.clean(df)
        stats = self.loader.statistics(df)
        logger.info("Dataset stats: %s", stats)
        self.loader.visualize(df)
        return df

    def get_techniques(self) -> Dict[str, Callable]:
        """Return mapping of technique name to prediction function."""
        few_shot_examples = gold_to_few_shot_examples("sentiment")

        return {
            "simple": lambda c, text: SimplePrompt.run(
                c, f"Is this review positive or negative? \"{text}\""
            ),
            "zero_shot": lambda c, text: ZeroShotPrompt.run(
                c, SENTIMENT_ZERO_SHOT, text
            ),
            "few_shot": lambda c, text: FewShotPrompt.run(
                c, SENTIMENT_TASK, few_shot_examples, text
            ),
            "chain_of_thought": lambda c, text: ChainOfThoughtPrompt.run(
                c, SENTIMENT_COT_TASK, text
            ),
            "role": lambda c, text: RolePrompt.run(
                c, self.TASK, text, role="sentiment"
            ),
            "self_consistency": lambda c, text: SelfConsistencyPrompt.run(
                c, self.TASK, text, n_samples=3
            ),
        }

    def run_all_techniques(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all prompting techniques and return comparison."""
        print_section("Evaluating Prompt Engineering Techniques")
        techniques = self.get_techniques()

        for name, predict_fn in techniques.items():
            logger.info("Running technique: %s", name)
            self.evaluator.evaluate_technique(name, df, predict_fn)

        comparison = self.evaluator.compare_all()
        print_results_table(
            "Sentiment Analysis — Technique Comparison",
            list(comparison.columns),
            comparison.values.tolist(),
        )
        self.evaluator.plot_comparison()
        return comparison

    def run_single(self, text: str, technique: str = "zero_shot") -> LLMResponse:
        """Run a single prediction with specified technique."""
        techniques = self.get_techniques()
        if technique not in techniques:
            raise ValueError(f"Unknown technique: {technique}")
        return techniques[technique](self.client, text)
