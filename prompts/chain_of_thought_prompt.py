"""
Chain of Thought (CoT) Prompt Technique
========================================

EXPLANATION:
Chain of Thought prompting encourages the model to break down complex
problems into intermediate reasoning steps before arriving at a final
answer. This significantly improves performance on reasoning tasks.

The magic phrase "Let's think step by step" triggers step-by-step reasoning.

WHEN TO USE:
- Complex reasoning tasks
- Multi-step classification requiring justification
- When interpretability of decisions matters

OUTPUT EXAMPLE:
    Input:  "Review: 'Great visuals but terrible dialogue.'"
    Output: "Step 1: 'Great visuals' is positive.
             Step 2: 'terrible dialogue' is negative.
             Step 3: Mixed but 'terrible' is stronger.
             Final Answer: negative"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llms.base_client import BaseLLMClient, LLMResponse


class ChainOfThoughtPrompt:
    """Step-by-step reasoning before final answer."""

    TECHNIQUE = "chain_of_thought"

    @staticmethod
    def build(task_description: str, input_text: str) -> str:
        """Build a CoT prompt with step-by-step instruction."""
        return (
            f"{task_description}\n\n"
            f'Text: "{input_text}"\n\n'
            "Let's think step by step:\n"
            "1. Identify key sentiment indicators in the text.\n"
            "2. Weigh positive vs negative signals.\n"
            "3. Determine the overall sentiment.\n\n"
            "Provide your reasoning, then state your final answer as "
            "either 'positive' or 'negative' on the last line."
        )

    @staticmethod
    def run(
        client: "BaseLLMClient",
        task_description: str,
        input_text: str,
    ) -> "LLMResponse":
        """Execute chain-of-thought prompt."""
        prompt = ChainOfThoughtPrompt.build(task_description, input_text)
        system = (
            "You are a careful analyst. Always show your reasoning step by step "
            "before giving a final answer."
        )
        return client.simple_prompt(prompt, system=system)

    @staticmethod
    def extract_answer(response_text: str) -> str:
        """Extract final classification from CoT response."""
        lines = response_text.strip().split("\n")
        for line in reversed(lines):
            lower = line.lower().strip()
            if "positive" in lower and "negative" not in lower:
                return "positive"
            if "negative" in lower and "positive" not in lower:
                return "negative"
            if lower in ("positive", "negative"):
                return lower
        return "unknown"


SENTIMENT_COT_TASK = (
    "Analyze the sentiment of the following movie review."
)


if __name__ == "__main__":
    from llms.openai_client import OpenAIClient

    client = OpenAIClient()
    response = ChainOfThoughtPrompt.run(
        client,
        SENTIMENT_COT_TASK,
        "The special effects were amazing but the story was confusing.",
    )
    print(response.content)
    print(f"\nExtracted: {ChainOfThoughtPrompt.extract_answer(response.content)}")
