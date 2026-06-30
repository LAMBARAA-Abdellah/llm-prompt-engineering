"""
Spam Detection Use Case
========================

Complete workflow for SMS spam detection using prompt engineering.
Repeats the full evaluation pipeline from sentiment analysis.
"""

from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from datasets.spam_loader import SpamLoader
from evaluation.evaluator import PromptEvaluator
from llms.base_client import BaseLLMClient, LLMResponse
from prompts.chain_of_thought_prompt import ChainOfThoughtPrompt
from prompts.few_shot_prompt import FewShotPrompt
from prompts.role_prompt import RolePrompt
from prompts.self_consistency_prompt import SelfConsistencyPrompt
from prompts.zero_shot_prompt import ZeroShotPrompt
from use_cases.gold_examples import gold_to_few_shot_examples
from utils.console import print_results_table, print_section
from utils.logger import get_logger

logger = get_logger(__name__)

SPAM_TASK = "Classify the following SMS message as 'ham' (legitimate) or 'spam'."
SPAM_ZERO_SHOT = (
    "Determine if this SMS message is spam or ham (legitimate). "
    "Spam includes promotional offers, phishing, and fraudulent messages."
)
SPAM_FEW_SHOT_EXAMPLES = [
    ("Hey, are we meeting tomorrow?", "ham"),
    ("FREE entry! Win £1000! Text WIN now!", "spam"),
    ("Can you send the report?", "ham"),
    ("URGENT: Verify your bank account now!", "spam"),
]


class SpamDetectionUseCase:
    """End-to-end spam detection with prompt engineering comparison."""

    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client
        self.loader = SpamLoader()
        self.evaluator = PromptEvaluator(client)

    def load_and_prepare(self, sample_size: int = 20) -> pd.DataFrame:
        """Load, clean, and sample SMS spam dataset."""
        print_section("Loading SMS Spam Dataset")
        df = self.loader.load(sample_size=sample_size)
        df = self.loader.clean(df)
        stats = self.loader.statistics(df)
        logger.info("Dataset stats: %s", stats)
        self.loader.visualize(df)
        return df

    def get_techniques(self) -> Dict[str, Callable]:
        """Return mapping of technique name to prediction function."""
        few_shot = gold_to_few_shot_examples("spam")

        return {
            "zero_shot": lambda c, text: ZeroShotPrompt.run(
                c, SPAM_ZERO_SHOT, text,
                output_format="Respond with only: ham or spam.",
            ),
            "few_shot": lambda c, text: FewShotPrompt.run(
                c, SPAM_TASK, few_shot, text,
                output_format="Respond with only: ham or spam.",
            ),
            "chain_of_thought": lambda c, text: ChainOfThoughtPrompt.run(
                c, "Analyze if this SMS is spam or ham.", text
            ),
            "role": lambda c, text: RolePrompt.run(
                c, SPAM_TASK, text, role="spam",
                output_format="Respond with only: ham or spam.",
            ),
            "self_consistency": lambda c, text: SelfConsistencyPrompt.run(
                c, SPAM_TASK, text, n_samples=3
            ),
        }

    def run_all_techniques(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all prompting techniques."""
        print_section("Evaluating Spam Detection Techniques")
        techniques = self.get_techniques()

        for name, predict_fn in techniques.items():
            self.evaluator.evaluate_technique(name, df, predict_fn)

        comparison = self.evaluator.compare_all()
        print_results_table(
            "Spam Detection — Technique Comparison",
            list(comparison.columns),
            comparison.values.tolist(),
        )

        from config.settings import get_settings
        save_path = str(get_settings().output_dir / "spam_technique_comparison.png")
        self.evaluator.plot_comparison(save_path=save_path)
        return comparison

    def run_single(self, text: str, technique: str = "zero_shot") -> LLMResponse:
        """Run single spam detection prediction."""
        techniques = self.get_techniques()
        return techniques[technique](self.client, text)
