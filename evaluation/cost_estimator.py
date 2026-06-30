"""
Cost estimation for LLM API usage.

Estimates costs based on token usage and provider pricing.
"""

from __future__ import annotations

from typing import Dict, List

from config.models import Provider, get_provider_config
from llms.base_client import LLMResponse
from utils.logger import get_logger

logger = get_logger(__name__)


class CostEstimator:
    """Estimate and track LLM API costs."""

    def __init__(self, provider: Provider | str, model: str) -> None:
        self.provider_config = get_provider_config(provider)
        self.model_config = self.provider_config.get_model(model)

    def estimate_single(self, response: LLMResponse) -> float:
        """Estimate cost for a single response."""
        cfg = self.model_config
        input_cost = (response.prompt_tokens / 1000) * cfg.cost_per_1k_input
        output_cost = (response.completion_tokens / 1000) * cfg.cost_per_1k_output
        return round(input_cost + output_cost, 6)

    def estimate_batch(self, responses: List[LLMResponse]) -> Dict[str, float]:
        """Estimate total cost for a batch of responses."""
        total_input = sum(r.prompt_tokens for r in responses)
        total_output = sum(r.completion_tokens for r in responses)
        cfg = self.model_config

        input_cost = (total_input / 1000) * cfg.cost_per_1k_input
        output_cost = (total_output / 1000) * cfg.cost_per_1k_output

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(input_cost + output_cost, 6),
        }

    @staticmethod
    def format_cost_report(costs: Dict[str, float]) -> str:
        """Format cost report for display."""
        lines = [
            "Cost Estimation Report",
            "=" * 40,
            f"  Input tokens:  {costs.get('total_input_tokens', 0):,}",
            f"  Output tokens: {costs.get('total_output_tokens', 0):,}",
            f"  Input cost:    ${costs.get('input_cost', 0):.6f}",
            f"  Output cost:   ${costs.get('output_cost', 0):.6f}",
            f"  Total cost:    ${costs.get('total_cost', 0):.6f}",
        ]
        return "\n".join(lines)
