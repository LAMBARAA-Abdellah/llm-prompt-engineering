"""Tests for prompt engineering modules."""

import pytest


def test_simple_prompt_build():
    from prompts.simple_prompt import SimplePrompt
    result = SimplePrompt.build("Hello world")
    assert result == "Hello world"


def test_zero_shot_prompt_build():
    from prompts.zero_shot_prompt import ZeroShotPrompt
    result = ZeroShotPrompt.build("Classify sentiment", "Great movie!", "Respond: positive or negative")
    assert "Great movie!" in result
    assert "Classify sentiment" in result


def test_few_shot_prompt_build():
    from prompts.few_shot_prompt import FewShotPrompt
    examples = [("Good film", "positive"), ("Bad film", "negative")]
    result = FewShotPrompt.build("Classify", examples, "Okay film")
    assert "Good film" in result
    assert "Bad film" in result
    assert "Okay film" in result


def test_cot_extract_answer_positive():
    from prompts.chain_of_thought_prompt import ChainOfThoughtPrompt
    text = "Step 1: positive words\nStep 2: overall positive\nFinal Answer: positive"
    assert ChainOfThoughtPrompt.extract_answer(text) == "positive"


def test_cot_extract_answer_negative():
    from prompts.chain_of_thought_prompt import ChainOfThoughtPrompt
    text = "The review is clearly negative.\nFinal Answer: negative"
    assert ChainOfThoughtPrompt.extract_answer(text) == "negative"


def test_gold_examples_sentiment():
    from use_cases.gold_examples import get_gold_examples
    examples = get_gold_examples("sentiment")
    assert len(examples) >= 5
    assert all("text" in ex and "label" in ex for ex in examples)


def test_gold_examples_spam():
    from use_cases.gold_examples import get_gold_examples
    examples = get_gold_examples("spam")
    assert len(examples) >= 5
    labels = {ex["label"] for ex in examples}
    assert "ham" in labels
    assert "spam" in labels
