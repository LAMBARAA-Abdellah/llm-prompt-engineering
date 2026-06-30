"""
Gold Examples for Prompt Engineering Evaluation
=================================================

WHY GOLD EXAMPLES ARE IMPORTANT:
--------------------------------
Gold examples (manually labeled, high-quality reference data) serve as the
ground truth for evaluating prompt engineering techniques. They are critical because:

1. **Benchmarking**: Without gold labels, we cannot measure accuracy, precision,
   recall, or F1 score. Gold examples provide the "correct answer" baseline.

2. **Few-Shot Learning**: Gold examples are directly used as demonstrations
   in few-shot prompting — their quality directly impacts model performance.

3. **Reproducibility**: Fixed gold examples ensure experiments are reproducible
   across different runs, models, and prompting techniques.

4. **Error Analysis**: Comparing predictions against gold labels reveals
   systematic errors and guides prompt refinement.

5. **Fair Comparison**: All prompting techniques are evaluated on the same
   gold set, ensuring fair comparison of accuracy, speed, and cost.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Sentiment Analysis Gold Examples (IMDB-style)
# ---------------------------------------------------------------------------

SENTIMENT_GOLD_EXAMPLES: List[Dict[str, str]] = [
    {
        "text": "An absolutely breathtaking film that redefines the genre.",
        "label": "positive",
        "rationale": "Strong positive adjectives: 'breathtaking', 'redefines'",
    },
    {
        "text": "The worst movie I have ever had the misfortune to watch.",
        "label": "negative",
        "rationale": "Explicit superlative negative: 'worst movie ever'",
    },
    {
        "text": "Decent performances but the script was incoherent.",
        "label": "negative",
        "rationale": "Mixed but 'incoherent script' dominates the sentiment",
    },
    {
        "text": "A touching story with brilliant character development.",
        "label": "positive",
        "rationale": "'Touching', 'brilliant' indicate strong positive sentiment",
    },
    {
        "text": "Not worth the ticket price. Save your money.",
        "label": "negative",
        "rationale": "Direct recommendation against watching",
    },
    {
        "text": "Masterful direction and a score that gives me chills.",
        "label": "positive",
        "rationale": "'Masterful' and emotional reaction indicate positive",
    },
    {
        "text": "Painfully slow with zero character development.",
        "label": "negative",
        "rationale": "'Painfully slow', 'zero development' are strong negatives",
    },
    {
        "text": "Exceeded all my expectations. A must-watch!",
        "label": "positive",
        "rationale": "'Exceeded expectations', 'must-watch' are positive signals",
    },
]

# ---------------------------------------------------------------------------
# Spam Detection Gold Examples (SMS-style)
# ---------------------------------------------------------------------------

SPAM_GOLD_EXAMPLES: List[Dict[str, str]] = [
    {
        "text": "Hey, are we still on for dinner at 7?",
        "label": "ham",
        "rationale": "Personal conversational message, no promotional content",
    },
    {
        "text": "FREE! Claim your £1000 prize now! Text WIN to 81234",
        "label": "spam",
        "rationale": "Promotional language, urgency, unknown shortcode",
    },
    {
        "text": "Can you send me the meeting notes from yesterday?",
        "label": "ham",
        "rationale": "Work-related legitimate request",
    },
    {
        "text": "URGENT: Your account suspended. Verify at http://fake-bank.com",
        "label": "spam",
        "rationale": "Phishing attempt with urgency and suspicious URL",
    },
    {
        "text": "Thanks for helping me move last weekend!",
        "label": "ham",
        "rationale": "Personal gratitude message",
    },
    {
        "text": "Congratulations! You've won a FREE iPhone 15. Click here to claim!",
        "label": "spam",
        "rationale": "Too-good-to-be-true offer with call to action",
    },
    {
        "text": "The train arrives at platform 3 in 10 minutes.",
        "label": "ham",
        "rationale": "Informational personal message",
    },
    {
        "text": "Lowest rates on loans! Call 0800123456 now for instant approval!",
        "label": "spam",
        "rationale": "Unsolicited financial promotion with urgency",
    },
]


def get_gold_examples(task: str = "sentiment") -> List[Dict[str, str]]:
    """Return gold examples for a given task."""
    if task == "sentiment":
        return SENTIMENT_GOLD_EXAMPLES
    if task == "spam":
        return SPAM_GOLD_EXAMPLES
    raise ValueError(f"Unknown task: {task}. Use 'sentiment' or 'spam'.")


def gold_to_few_shot_examples(task: str = "sentiment") -> list:
    """Convert gold examples to few-shot (input, output) tuples."""
    gold = get_gold_examples(task)
    return [(ex["text"], ex["label"]) for ex in gold[:4]]
