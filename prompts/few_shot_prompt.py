"""
Few-Shot Prompt Technique
==========================

EXPLANATION:
Few-Shot prompting provides the model with a small number of labeled
examples (typically 2-5) before the actual input. This helps the model
understand the expected input-output pattern without fine-tuning.

WHEN TO USE:
- When zero-shot accuracy is insufficient
- Tasks with specific output formats
- Domain-specific classification

OUTPUT EXAMPLE:
    Examples:
      "Great film!" -> positive
      "Awful experience." -> negative
    Input: "Mediocre at best." -> negative
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from llms.base_client import BaseLLMClient, LLMResponse


class FewShotPrompt:
    """Prompt with labeled examples to guide model behavior."""

    TECHNIQUE = "few_shot"

    @staticmethod
    def build(
        task_description: str,
        examples: List[Tuple[str, str]],
        input_text: str,
        output_format: str = "",
    ) -> str:
        """
        Build a few-shot prompt with examples.

        Args:
            task_description: Task instruction.
            examples: List of (input, output) example pairs.
            input_text: The text to classify/process.
            output_format: Optional output constraint.
        """
        lines = [task_description, "", "Examples:"]
        for i, (inp, out) in enumerate(examples, 1):
            lines.append(f'  Example {i}:')
            lines.append(f'    Input: "{inp}"')
            lines.append(f'    Output: {out}')
        lines.append("")
        lines.append(f'Now classify this:')
        lines.append(f'  Input: "{input_text}"')
        lines.append(f'  Output:')
        if output_format:
            lines.append(f"\n{output_format}")
        return "\n".join(lines)

    @staticmethod
    def run(
        client: "BaseLLMClient",
        task_description: str,
        examples: List[Tuple[str, str]],
        input_text: str,
        output_format: str = "Respond with only: positive or negative.",
    ) -> "LLMResponse":
        """Execute few-shot prompt."""
        prompt = FewShotPrompt.build(task_description, examples, input_text, output_format)
        system = "You are an NLP expert. Learn from the examples and classify accurately."
        return client.simple_prompt(prompt, system=system)


# Default sentiment examples
SENTIMENT_FEW_SHOT_EXAMPLES = [
    ("This movie was absolutely brilliant!", "positive"),
    ("Worst film I've ever seen.", "negative"),
    ("The cinematography was stunning.", "positive"),
    ("Boring and predictable storyline.", "negative"),
]

SENTIMENT_TASK = "Classify the sentiment of movie reviews as 'positive' or 'negative'."


if __name__ == "__main__":
    from llms.openai_client import OpenAIClient

    client = OpenAIClient()
    response = FewShotPrompt.run(
        client,
        SENTIMENT_TASK,
        SENTIMENT_FEW_SHOT_EXAMPLES,
        "It was okay, nothing special.",
    )
    print(f"Sentiment: {response.content.strip()}")
