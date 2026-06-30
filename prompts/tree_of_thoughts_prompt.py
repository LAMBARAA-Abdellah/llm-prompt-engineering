"""
Tree of Thoughts (ToT) Prompt Technique — BONUS
================================================

EXPLANATION:
Tree of Thoughts extends Chain of Thought by exploring multiple reasoning
branches and evaluating each path before selecting the best answer.
It simulates a search tree where each node is a partial solution.

Process:
1. Generate multiple initial thoughts (branches)
2. Evaluate each branch's promise
3. Expand the most promising branches
4. Select the best final answer

WHEN TO USE:
- Complex multi-step reasoning
- Problems with multiple valid approaches
- When single-path CoT fails

OUTPUT EXAMPLE:
    Branch A: Focus on "amazing" -> positive (score: 0.8)
    Branch B: Focus on "but boring" -> negative (score: 0.6)
    Branch C: Overall tone analysis -> positive (score: 0.9)
    Selected: positive (Branch C)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from llms.base_client import BaseLLMClient, LLMResponse


class TreeOfThoughtsPrompt:
    """Multi-branch reasoning with evaluation and selection."""

    TECHNIQUE = "tree_of_thoughts"

    @staticmethod
    def build_generation_prompt(task_description: str, input_text: str, n_branches: int = 3) -> str:
        """Generate multiple reasoning branches."""
        return (
            f"{task_description}\n\n"
            f'Text: "{input_text}"\n\n'
            f"Generate {n_branches} different reasoning paths to classify this text.\n"
            "For each path:\n"
            "  - Path N: [reasoning] -> [positive/negative] (confidence: 0.0-1.0)\n\n"
            "Consider different aspects: word choice, overall tone, context, negations."
        )

    @staticmethod
    def build_selection_prompt(branches_text: str) -> str:
        """Evaluate branches and select the best answer."""
        return (
            f"Here are multiple reasoning paths:\n\n{branches_text}\n\n"
            "Evaluate each path's reasoning quality and confidence score.\n"
            "Select the best path and respond with only: positive or negative."
        )

    @staticmethod
    def run(
        client: "BaseLLMClient",
        task_description: str,
        input_text: str,
        n_branches: int = 3,
    ) -> "LLMResponse":
        """Execute Tree of Thoughts prompting."""
        # Stage 1: Generate branches
        gen_prompt = TreeOfThoughtsPrompt.build_generation_prompt(
            task_description, input_text, n_branches
        )
        branches_response = client.simple_prompt(
            gen_prompt,
            system="You are an expert reasoning engine. Explore multiple perspectives.",
        )

        # Stage 2: Select best branch
        sel_prompt = TreeOfThoughtsPrompt.build_selection_prompt(branches_response.content)
        final_response = client.simple_prompt(
            sel_prompt,
            system="You are a critical evaluator. Choose the most sound reasoning path.",
        )

        total_time = branches_response.execution_time + final_response.execution_time
        total_tokens = branches_response.total_tokens + final_response.total_tokens

        from llms.base_client import LLMResponse

        return LLMResponse(
            content=final_response.content.strip(),
            model=client.model,
            provider=client.provider_name,
            execution_time=total_time,
            total_tokens=total_tokens,
            metadata={
                "branches": branches_response.content,
                "n_branches": n_branches,
            },
        )

    @staticmethod
    def extract_answer(response_text: str) -> str:
        """Extract classification from ToT response."""
        text = response_text.lower().strip()
        if "positive" in text and "negative" not in text:
            return "positive"
        if "negative" in text and "positive" not in text:
            return "negative"
        return text.split()[-1] if text else "unknown"


if __name__ == "__main__":
    from llms.openai_client import OpenAIClient

    client = OpenAIClient()
    response = TreeOfThoughtsPrompt.run(
        client,
        "Classify the sentiment of this review.",
        "Visually stunning but the pacing was painfully slow.",
    )
    print(f"ToT Result: {TreeOfThoughtsPrompt.extract_answer(response.content)}")
