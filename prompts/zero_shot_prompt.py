"""
Zero-Shot Prompt Technique
===========================

EXPLANATION:
Zero-Shot prompting asks the model to perform a task WITHOUT providing
any examples. The model must rely on its pre-trained understanding of
the task description. A clear task definition and output format
instruction improves results.

WHEN TO USE:
- Classification tasks with well-known categories
- When labeled examples are unavailable
- Quick prototyping of NLP pipelines

OUTPUT EXAMPLE:
    Input:  "Classify the sentiment: 'This movie was fantastic!'"
    Output: "positive"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llms.base_client import BaseLLMClient, LLMResponse


class ZeroShotPrompt:
    """Task instruction without examples — relies on pre-trained knowledge."""

    TECHNIQUE = "zero_shot"

    @staticmethod
    def build(task_description: str, input_text: str, output_format: str = "") -> str:
        """
        Build a zero-shot classification/generation prompt.

        Args:
            task_description: Clear description of the task.
            input_text: The text to process.
            output_format: Optional format constraint (e.g., "Respond with only: positive or negative").
        """
        prompt = f"{task_description}\n\nText: \"{input_text}\""
        if output_format:
            prompt += f"\n\n{output_format}"
        return prompt

    @staticmethod
    def run(
        client: "BaseLLMClient",
        task_description: str,
        input_text: str,
        output_format: str = "Respond with only one word: positive or negative.",
    ) -> "LLMResponse":
        """Execute zero-shot prompt."""
        prompt = ZeroShotPrompt.build(task_description, input_text, output_format)
        system = "You are a precise NLP classifier. Follow instructions exactly."
        return client.simple_prompt(prompt, system=system)


# --- Sentiment Analysis Zero-Shot Template ---
SENTIMENT_ZERO_SHOT = (
    "Classify the sentiment of the following movie review as either "
    "'positive' or 'negative'."
)


if __name__ == "__main__":
    from llms.openai_client import OpenAIClient

    client = OpenAIClient()
    response = ZeroShotPrompt.run(
        client,
        SENTIMENT_ZERO_SHOT,
        "The acting was terrible and the plot made no sense.",
    )
    print(f"Sentiment: {response.content.strip()}")
