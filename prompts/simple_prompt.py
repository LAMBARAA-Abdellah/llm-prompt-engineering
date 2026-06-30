"""
Simple Prompt Technique
========================

EXPLANATION:
A Simple Prompt is the most basic form of interaction with an LLM.
You provide a direct instruction or question without any additional
context, examples, or structured formatting. The model relies entirely
on its pre-trained knowledge to generate a response.

WHEN TO USE:
- Quick factual queries
- Simple text generation tasks
- Baseline comparisons for more advanced techniques

OUTPUT EXAMPLE:
    Input:  "What is the capital of France?"
    Output: "The capital of France is Paris."
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llms.base_client import BaseLLMClient, LLMResponse


class SimplePrompt:
    """Direct, unstructured prompt with no additional context."""

    TECHNIQUE = "simple"

    @staticmethod
    def build(user_input: str) -> str:
        """Build a simple prompt from user input."""
        return user_input

    @staticmethod
    def run(client: "BaseLLMClient", user_input: str) -> "LLMResponse":
        """
        Execute a simple prompt against an LLM client.

        Args:
            client: Any LLM client implementing BaseLLMClient.
            user_input: The raw user question or instruction.

        Returns:
            LLMResponse with the model's answer.
        """
        prompt = SimplePrompt.build(user_input)
        return client.simple_prompt(prompt)


# --- Demo ---
if __name__ == "__main__":
    from llms.openai_client import OpenAIClient

    client = OpenAIClient()
    response = SimplePrompt.run(client, "Explain photosynthesis in one sentence.")
    print(f"Response: {response.content}")
    print(f"Time: {response.execution_time:.2f}s | Tokens: {response.total_tokens}")
