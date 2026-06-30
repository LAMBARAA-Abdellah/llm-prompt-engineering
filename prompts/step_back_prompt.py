"""
Step-Back Prompt Technique
===========================

EXPLANATION:
Step-Back Prompting first asks the model to derive high-level principles
or abstractions relevant to the task, then applies those principles to
solve the specific problem. This reduces errors on complex tasks.

Process:
1. Ask: "What are the key principles for [task]?"
2. Use those principles to analyze the specific input

WHEN TO USE:
- Complex domain tasks
- When the model makes superficial errors
- Tasks requiring deep understanding before classification

OUTPUT EXAMPLE:
    Step 1 Principles: "Sentiment depends on emotional tone, not factual content."
    Step 2 Analysis: "Despite mentioning flaws, overall tone is appreciative."
    Final: positive
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llms.base_client import BaseLLMClient, LLMResponse


class StepBackPrompt:
    """Two-stage prompting: derive principles, then apply them."""

    TECHNIQUE = "step_back"

    @staticmethod
    def build_principles_prompt(domain: str) -> str:
        """Stage 1: Ask for high-level principles."""
        return (
            f"What are the key principles and guidelines for {domain}? "
            "List 3-5 concise principles."
        )

    @staticmethod
    def build_application_prompt(
        principles: str,
        task_description: str,
        input_text: str,
    ) -> str:
        """Stage 2: Apply principles to specific input."""
        return (
            f"Based on these principles:\n{principles}\n\n"
            f"{task_description}\n\n"
            f'Text: "{input_text}"\n\n'
            "Apply the principles above and respond with only: positive or negative."
        )

    @staticmethod
    def run(
        client: "BaseLLMClient",
        task_description: str,
        input_text: str,
        domain: str = "sentiment analysis of movie reviews",
    ) -> "LLMResponse":
        """
        Execute two-stage step-back prompt.

        Returns combined response with principles and final classification.
        """
        # Stage 1: Derive principles
        principles_prompt = StepBackPrompt.build_principles_prompt(domain)
        principles_response = client.simple_prompt(
            principles_prompt,
            system="You are a domain expert. Be concise and precise.",
        )

        # Stage 2: Apply principles
        application_prompt = StepBackPrompt.build_application_prompt(
            principles_response.content,
            task_description,
            input_text,
        )
        final_response = client.simple_prompt(
            application_prompt,
            system="You are a precise classifier applying expert principles.",
        )

        total_time = principles_response.execution_time + final_response.execution_time
        total_tokens = principles_response.total_tokens + final_response.total_tokens

        from llms.base_client import LLMResponse

        return LLMResponse(
            content=final_response.content,
            model=client.model,
            provider=client.provider_name,
            execution_time=total_time,
            total_tokens=total_tokens,
            metadata={
                "principles": principles_response.content,
                "stage1_tokens": principles_response.total_tokens,
                "stage2_tokens": final_response.total_tokens,
            },
        )


if __name__ == "__main__":
    from llms.openai_client import OpenAIClient

    client = OpenAIClient()
    response = StepBackPrompt.run(
        client,
        "Classify the sentiment.",
        "Not bad, could have been better though.",
    )
    print(f"Result: {response.content.strip()}")
    print(f"Principles used:\n{response.metadata.get('principles', '')[:200]}...")
