#!/usr/bin/env python3
"""
Generate final PDF report for the Prompt Engineering project.

Creates a comprehensive academic report covering all project sections.
Run: python docs/generate_report.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fpdf import FPDF


class ReportPDF(FPDF):
    """Custom PDF with header and footer."""

    def header(self) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Prompt Engineering with Python using LLMs", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 16, 200, 16)
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(0, 51, 102)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def section_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0, 76, 153)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def bullet_list(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 11)
        self.set_text_color(30, 30, 30)
        for item in items:
            self.cell(8)
            self.multi_cell(0, 6, f"- {item}")
        self.ln(3)


def generate_report() -> Path:
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Title Page
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "Prompt Engineering with Python", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, "using Large Language Models (LLMs)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Academic Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "Master BDCC", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "Universite Mohammed VI Polytechnique", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%B %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")

    # 1. Introduction
    pdf.add_page()
    pdf.chapter_title("1. Introduction")
    pdf.body_text(
        "Prompt Engineering has emerged as a critical discipline in the field of Artificial "
        "Intelligence and Natural Language Processing. As Large Language Models (LLMs) become "
        "increasingly powerful and accessible, the ability to craft effective prompts determines "
        "the quality, accuracy, and reliability of model outputs. This project presents a "
        "comprehensive Python application that implements, compares, and evaluates multiple "
        "prompt engineering techniques across real-world NLP tasks."
    )
    pdf.body_text(
        "The project supports three LLM providers (OpenAI, Ollama, Groq), implements eight "
        "prompting strategies, and evaluates them on sentiment analysis and spam detection "
        "using publicly available datasets. Additionally, multimodal prompting capabilities "
        "are demonstrated through text-to-image and image-to-image generation."
    )

    # 2. Background
    pdf.section_title("2. Background")
    pdf.body_text(
        "Large Language Models are transformer-based neural networks trained on massive text "
        "corpora. Models like GPT-4, Llama 3, and Mixtral can perform diverse NLP tasks "
        "through natural language instructions without task-specific fine-tuning. This "
        "capability, known as zero-shot or in-context learning, forms the foundation of "
        "prompt engineering."
    )
    pdf.body_text(
        "Prompt engineering encompasses techniques for designing inputs that guide LLMs toward "
        "desired outputs. Unlike traditional ML pipelines that require labeled data and model "
        "training, prompt engineering leverages pre-trained knowledge through carefully "
        "crafted text instructions, examples, and reasoning structures."
    )

    # 3. Prompt Engineering
    pdf.add_page()
    pdf.chapter_title("3. Prompt Engineering Techniques")
    sections = [
        ("3.1 Simple Prompt", "The most basic approach: direct instruction without context or examples."),
        ("3.2 Zero-Shot Prompt", "Task description without examples, relying on pre-trained knowledge."),
        ("3.3 Few-Shot Prompt", "Provides 2-5 labeled examples before the input to guide output format."),
        ("3.4 Chain of Thought", "Encourages step-by-step reasoning before the final answer."),
        ("3.5 Self-Consistency", "Generates multiple responses and selects via majority voting."),
        ("3.6 Role Prompt", "Assigns an expert persona through the system message."),
        ("3.7 Step-Back Prompt", "First derives high-level principles, then applies them."),
        ("3.8 Tree of Thoughts", "Explores multiple reasoning branches and selects the best."),
    ]
    for title, desc in sections:
        pdf.section_title(title)
        pdf.body_text(desc)

    # 4. LLMs
    pdf.add_page()
    pdf.chapter_title("4. Large Language Models")
    pdf.section_title("4.1 OpenAI")
    pdf.body_text("GPT-4o-mini for chat completions and DALL-E 3 for image generation. Cloud-based with pay-per-token pricing.")
    pdf.section_title("4.2 Ollama")
    pdf.body_text("Local inference supporting Llama3, Mistral, and Gemma. Zero API cost, requires local hardware.")
    pdf.section_title("4.3 Groq")
    pdf.body_text("Fast cloud inference for Llama 3.3 70B and Mixtral. Optimized hardware for low-latency responses.")

    # 5. Methodology
    pdf.add_page()
    pdf.chapter_title("5. Methodology")
    pdf.body_text("The evaluation follows a systematic pipeline:")
    pdf.bullet_list([
        "Load and preprocess public datasets (IMDB, SMS Spam)",
        "Apply each prompting technique to generate predictions",
        "Compare predictions against manually labeled gold examples",
        "Compute Accuracy, Precision, Recall, and F1 Score",
        "Measure execution time and estimate API costs",
        "Generate comparison visualizations",
    ])

    # 6. Implementation
    pdf.section_title("6. Implementation")
    pdf.body_text(
        "The project follows clean architecture with Python 3.12, type hints, and PEP8 "
        "compliance. An abstract BaseLLMClient defines the interface for all providers. "
        "Each prompting technique is a standalone module. The PromptEvaluator orchestrates "
        "batch evaluation with progress bars and automatic metric computation."
    )

    # 7. Experiments
    pdf.add_page()
    pdf.chapter_title("7. Experiments")
    pdf.section_title("7.1 Sentiment Analysis (IMDB)")
    pdf.body_text("20 movie reviews classified using 6 prompting techniques. Gold examples provide ground truth.")
    pdf.section_title("7.2 Spam Detection (SMS)")
    pdf.body_text("20 SMS messages classified as ham or spam using 5 prompting techniques.")

    # 8. Evaluation
    pdf.section_title("8. Evaluation")
    pdf.bullet_list([
        "Accuracy: Overall correct predictions / total predictions",
        "Precision: True positives / (True positives + False positives)",
        "Recall: True positives / (True positives + False negatives)",
        "F1 Score: Harmonic mean of precision and recall",
        "Confusion Matrix: Visual breakdown of prediction errors",
        "Execution Time: Average seconds per prediction",
        "Cost Estimation: Token usage multiplied by provider pricing",
    ])

    # 9. Comparison
    pdf.add_page()
    pdf.chapter_title("9. Comparison & Results")
    pdf.body_text(
        "Results vary by technique, model, and task. Generally, Few-Shot and Role Prompting "
        "achieve the highest accuracy by providing clear context. Zero-Shot offers a strong "
        "baseline with minimal token cost. Chain of Thought improves interpretability at the "
        "cost of additional tokens. Self-Consistency reduces prediction variance through "
        "majority voting but increases cost proportionally to sample count."
    )
    pdf.body_text(
        "Ollama provides free local inference suitable for development and prototyping. "
        "OpenAI and Groq offer higher accuracy with cloud-based models at per-token costs."
    )

    # 10. Conclusion
    pdf.section_title("10. Conclusion")
    pdf.body_text(
        "This project delivers a complete, modular framework for prompt engineering research "
        "and education. Eight techniques are implemented and compared on two NLP tasks with "
        "quantitative metrics. The architecture supports easy extension with new providers, "
        "techniques, and use cases."
    )

    # 11. Future Work
    pdf.section_title("11. Future Work")
    pdf.bullet_list([
        "Compare prompt engineering vs fine-tuning on same tasks",
        "Integrate Retrieval-Augmented Generation (RAG)",
        "Automated prompt optimization with DSPy",
        "Multi-language dataset support",
        "Real-time evaluation dashboard",
        "LangChain/LlamaIndex integration",
    ])

    output_path = PROJECT_ROOT / "outputs" / "Prompt_Engineering_Report.pdf"
    pdf.output(str(output_path))
    print(f"Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_report()
