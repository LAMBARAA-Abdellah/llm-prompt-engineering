#!/usr/bin/env python3
"""
Prompt Engineering with Python using Large Language Models (LLMs)
==================================================================

Main entry point for the academic project.

Usage:
    python main.py                          # Interactive menu
    python main.py --demo prompts           # Run prompt demos
    python main.py --demo sentiment         # Sentiment analysis workflow
    python main.py --demo spam              # Spam detection workflow
    python main.py --demo text2img          # Text-to-image generation
    python main.py --demo img2img           # Image-to-image transformation
    python main.py --provider openai        # Choose LLM provider
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from utils.logger import setup_logging, get_logger
from utils.console import print_banner, print_section

logger = get_logger(__name__)


def get_client(provider: str):
    """Factory function to create LLM client by provider name."""
    if provider == "openai":
        from llms.openai_client import OpenAIClient
        return OpenAIClient()
    elif provider == "ollama":
        from llms.ollama_client import OllamaClient
        return OllamaClient()
    elif provider == "groq":
        from llms.groq_client import GroqClient
        return GroqClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Use: openai, ollama, groq")


def demo_prompts(client) -> None:
    """Demonstrate all prompt engineering techniques."""
    print_section("Prompt Engineering Techniques Demo")

    from prompts.simple_prompt import SimplePrompt
    from prompts.zero_shot_prompt import ZeroShotPrompt, SENTIMENT_ZERO_SHOT
    from prompts.few_shot_prompt import FewShotPrompt, SENTIMENT_TASK, SENTIMENT_FEW_SHOT_EXAMPLES
    from prompts.chain_of_thought_prompt import ChainOfThoughtPrompt, SENTIMENT_COT_TASK
    from prompts.role_prompt import RolePrompt
    from prompts.step_back_prompt import StepBackPrompt
    from prompts.tree_of_thoughts_prompt import TreeOfThoughtsPrompt

    test_text = "The movie had stunning visuals but the plot was confusing."

    techniques = [
        ("Simple Prompt", lambda: SimplePrompt.run(client, f"Sentiment of: {test_text}")),
        ("Zero-Shot", lambda: ZeroShotPrompt.run(client, SENTIMENT_ZERO_SHOT, test_text)),
        ("Few-Shot", lambda: FewShotPrompt.run(client, SENTIMENT_TASK, SENTIMENT_FEW_SHOT_EXAMPLES, test_text)),
        ("Chain of Thought", lambda: ChainOfThoughtPrompt.run(client, SENTIMENT_COT_TASK, test_text)),
        ("Role Prompt", lambda: RolePrompt.run(client, "Classify sentiment.", test_text)),
        ("Step-Back", lambda: StepBackPrompt.run(client, "Classify sentiment.", test_text)),
        ("Tree of Thoughts", lambda: TreeOfThoughtsPrompt.run(client, "Classify sentiment.", test_text)),
    ]

    for name, fn in techniques:
        print_section(name)
        try:
            response = fn()
            print(f"  Response: {response.content[:200]}")
            print(f"  Time: {response.execution_time:.2f}s | Tokens: {response.total_tokens}")
        except Exception as exc:
            print(f"  Error: {exc}")


def demo_sentiment(client) -> None:
    """Run sentiment analysis use case."""
    from use_cases.sentiment_analysis import SentimentAnalysisUseCase

    settings = get_settings()
    use_case = SentimentAnalysisUseCase(client)
    df = use_case.load_and_prepare(sample_size=min(settings.eval_sample_size, 10))
    comparison = use_case.run_all_techniques(df)

    output_path = settings.output_dir / "sentiment_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


def demo_spam(client) -> None:
    """Run spam detection use case."""
    from use_cases.spam_detection import SpamDetectionUseCase

    settings = get_settings()
    use_case = SpamDetectionUseCase(client)
    df = use_case.load_and_prepare(sample_size=min(settings.eval_sample_size, 10))
    comparison = use_case.run_all_techniques(df)

    output_path = settings.output_dir / "spam_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


def demo_text2img() -> None:
    """Run text-to-image generation."""
    from use_cases.text_to_image import TextToImageUseCase

    use_case = TextToImageUseCase()
    path = use_case.generate(
        "A professional academic presentation about prompt engineering with LLMs"
    )
    print(f"Image saved to: {path}")


def demo_img2img() -> None:
    """Run image-to-image transformation."""
    from use_cases.image_to_image import ImageToImageUseCase

    use_case = ImageToImageUseCase()
    path = use_case.run_demo()
    print(f"Transformed image saved to: {path}")


def interactive_menu(client) -> None:
    """Display interactive CLI menu."""
    while True:
        print_section("Main Menu")
        print("  1. Demo Prompt Techniques")
        print("  2. Sentiment Analysis (IMDB)")
        print("  3. Spam Detection (SMS)")
        print("  4. Text-to-Image Generation")
        print("  5. Image-to-Image Transformation")
        print("  6. Exit")

        choice = input("\nSelect option (1-6): ").strip()
        actions = {
            "1": lambda: demo_prompts(client),
            "2": lambda: demo_sentiment(client),
            "3": lambda: demo_spam(client),
            "4": demo_text2img,
            "5": demo_img2img,
            "6": lambda: sys.exit(0),
        }
        action = actions.get(choice)
        if action:
            action()
        else:
            print("Invalid choice. Please select 1-6.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prompt Engineering with LLMs — Academic Project"
    )
    parser.add_argument(
        "--provider", choices=["openai", "ollama", "groq"], default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--demo", choices=["prompts", "sentiment", "spam", "text2img", "img2img"],
        help="Run a specific demo non-interactively",
    )
    args = parser.parse_args()

    setup_logging()
    print_banner(
        "Prompt Engineering with Python using LLMs",
        "Master BDCC — Université Mohammed VI Polytechnique",
    )

    settings = get_settings()
    logger.info("Project root: %s", settings.project_root)

    # Image demos don't need chat client
    if args.demo == "text2img":
        demo_text2img()
        return
    if args.demo == "img2img":
        demo_img2img()
        return

    try:
        client = get_client(args.provider)
        logger.info("Using provider: %s | model: %s", args.provider, client.model)
    except Exception as exc:
        logger.error("Failed to initialize client: %s", exc)
        print(f"\nError: {exc}")
        print("Ensure your API keys are set in .env (copy from .env.example)")
        sys.exit(1)

    demos = {
        "prompts": lambda: demo_prompts(client),
        "sentiment": lambda: demo_sentiment(client),
        "spam": lambda: demo_spam(client),
    }

    if args.demo:
        demos[args.demo]()
    else:
        interactive_menu(client)


if __name__ == "__main__":
    main()
