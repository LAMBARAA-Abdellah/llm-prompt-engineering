"""
Self-Consistency Prompt Technique
==================================

EXPLANATION:
Self-Consistency generates multiple reasoning paths for the same problem
and selects the most frequent answer through majority voting. This reduces
variance and improves reliability on complex tasks.

Process:
1. Generate N responses with temperature > 0
2. Extract the answer from each response
3. Return the majority vote

WHEN TO USE:
- High-stakes classification
- When single-shot accuracy is inconsistent
- Reasoning tasks where multiple valid paths exist

OUTPUT EXAMPLE:
    Run 1: positive
    Run 2: positive
    Run 3: negative
    Majority Vote: positive
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from llms.base_client import BaseLLMClient, LLMResponse


class SelfConsistencyPrompt:
    """Majority voting over multiple LLM responses."""

    TECHNIQUE = "self_consistency"

    @staticmethod
    def build(task_description: str, input_text: str) -> str:
        """Build prompt for self-consistency sampling."""
        return (
            f"{task_description}\n\n"
            f'Text: "{input_text}"\n\n'
            "Analyze carefully and respond with only: positive or negative."
        )

    @staticmethod
    def run(
        client: "BaseLLMClient",
        task_description: str,
        input_text: str,
        n_samples: int = 5,
        temperature: float = 0.7,
        extract_fn: Callable[[str], str] | None = None,
    ) -> "LLMResponse":
        """
        Execute self-consistency with majority voting.

        Args:
            client: LLM client.
            task_description: Task instruction.
            input_text: Text to classify.
            n_samples: Number of samples to generate.
            temperature: Sampling temperature for diversity.
            extract_fn: Function to extract label from response.

        Returns:
            LLMResponse with majority-voted answer in content.
        """
        prompt = SelfConsistencyPrompt.build(task_description, input_text)
        system = "You are a sentiment classifier. Be precise."

        answers: List[str] = []
        total_time = 0.0
        total_tokens = 0

        for _ in range(n_samples):
            response = client.simple_prompt(prompt, system=system)
            # Override temperature via chat for diversity
            response = client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=temperature,
            )
            raw = response.content.strip().lower()
            label = extract_fn(raw) if extract_fn else SelfConsistencyPrompt._extract_label(raw)
            answers.append(label)
            total_time += response.execution_time
            total_tokens += response.total_tokens

        vote = Counter(answers).most_common(1)[0][0]

        from llms.base_client import LLMResponse

        return LLMResponse(
            content=vote,
            model=client.model,
            provider=client.provider_name,
            execution_time=total_time,
            total_tokens=total_tokens,
            metadata={"votes": dict(Counter(answers)), "n_samples": n_samples},
        )

    @staticmethod
    def _extract_label(text: str) -> str:
        """Extract positive/negative from response text."""
        text = text.lower()
        if "positive" in text and "negative" not in text:
            return "positive"
        if "negative" in text and "positive" not in text:
            return "negative"
        return text.split()[0] if text else "unknown"


if __name__ == "__main__":
    from llms.openai_client import OpenAIClient

    client = OpenAIClient()
    result = SelfConsistencyPrompt.run(
        client,
        "Classify sentiment as positive or negative.",
        "An enjoyable evening at the cinema.",
        n_samples=3,
    )
    print(f"Majority vote: {result.content}")
    print(f"Votes: {result.metadata.get('votes')}")
