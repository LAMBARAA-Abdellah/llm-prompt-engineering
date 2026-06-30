"""
Role Prompt Technique
=====================

EXPLANATION:
Role Prompting assigns a specific persona or expert role to the LLM
via the system message. This shapes the tone, vocabulary, and reasoning
style of responses, leading to more domain-appropriate outputs.

WHEN TO USE:
- Domain-specific analysis (legal, medical, film criticism)
- When consistent expert tone is needed
- Specialized classification tasks

OUTPUT EXAMPLE:
    Role: "You are an expert film critic with 20 years of experience."
    Input: "The movie was visually stunning."
    Output: "As a seasoned critic, I'd classify this as positive — the reviewer
             highlights visual excellence, a key quality marker."
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llms.base_client import BaseLLMClient, LLMResponse


class RolePrompt:
    """Assign an expert persona to guide LLM behavior."""

    TECHNIQUE = "role"

    DEFAULT_ROLES = {
        "sentiment": (
            "You are an expert NLP sentiment analyst with deep expertise in "
            "movie review classification. You have analyzed over 100,000 reviews "
            "and understand subtle emotional cues in text."
        ),
        "spam": (
            "You are a cybersecurity expert specializing in SMS spam detection. "
            "You can identify fraudulent messages, phishing attempts, and "
            "promotional spam with high accuracy."
        ),
        "general": (
            "You are a helpful, knowledgeable AI assistant with expertise "
            "across multiple domains."
        ),
    }

    @staticmethod
    def build(task_description: str, input_text: str) -> str:
        """Build the user prompt portion."""
        return f"{task_description}\n\nText: \"{input_text}\"\n\nProvide your classification."

    @staticmethod
    def run(
        client: "BaseLLMClient",
        task_description: str,
        input_text: str,
        role: str = "sentiment",
        output_format: str = "Respond with only: positive or negative.",
    ) -> "LLMResponse":
        """
        Execute role-based prompt.

        Args:
            client: LLM client.
            task_description: Task to perform.
            input_text: Input text.
            role: Key from DEFAULT_ROLES or custom role string.
            output_format: Output constraint.
        """
        system_role = RolePrompt.DEFAULT_ROLES.get(role, role)
        prompt = RolePrompt.build(task_description, input_text)
        prompt += f"\n\n{output_format}"
        return client.simple_prompt(prompt, system=system_role)


if __name__ == "__main__":
    from llms.openai_client import OpenAIClient

    client = OpenAIClient()
    response = RolePrompt.run(
        client,
        "Classify the sentiment of this movie review.",
        "A masterpiece of modern cinema!",
        role="sentiment",
    )
    print(f"Classification: {response.content.strip()}")
