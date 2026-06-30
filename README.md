# Prompt Engineering with Python using Large Language Models (LLMs)

**Master BDCC — Université Mohammed VI Polytechnique**

A complete academic project demonstrating prompt engineering techniques using OpenAI, Ollama, and Groq LLM providers. Compares 8 prompting strategies on real NLP tasks with quantitative evaluation.

---

## Project Overview

This project implements a modular, production-ready Python application that:

- Supports **3 LLM providers**: OpenAI (GPT-4o-mini, DALL-E), Ollama (Llama3, Mistral, Gemma), Groq (Llama 3.3 70B, Mixtral)
- Implements **8 prompt engineering techniques**: Simple, Zero-Shot, Few-Shot, Chain of Thought, Self-Consistency, Role, Step-Back, Tree of Thoughts
- Evaluates on **2 NLP use cases**: Sentiment Analysis (IMDB) and Spam Detection (SMS)
- Demonstrates **multimodal prompting**: Text-to-Image and Image-to-Image with DALL-E
- Computes **comprehensive metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, Execution Time, Cost

---

## Architecture

```
PromptEngineeringProject/
├── main.py                 # CLI entry point
├── config/                 # Settings & model registry (.env based)
├── llms/                   # OpenAI, Ollama, Groq clients
├── prompts/                # 8 prompting technique modules
├── datasets/               # IMDB & SMS Spam loaders
├── evaluation/             # Metrics, evaluator, cost estimator
├── use_cases/              # Sentiment, spam, image workflows
├── utils/                  # Logging & console output
├── notebooks/              # Jupyter analysis notebook
├── tests/                  # pytest test suite
├── docs/                   # Architecture, workflow, generators
├── images/                 # Generated images
└── outputs/                # Results, charts, reports
```

See [docs/architecture.md](docs/architecture.md) and [docs/workflow.md](docs/workflow.md) for detailed diagrams.

---

## Installation

### Prerequisites

- Python 3.12+
- OpenAI API key (for OpenAI and DALL-E features)
- Groq API key (optional, for Groq provider)
- Ollama installed locally (optional, for local inference)

### Setup

```bash
# Navigate to project directory
cd "C:\Users\abdel\OneDrive - Université Mohammed VI Polytechnique\Documents\Master BDCC\PromptEngineeringProject"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your API keys
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `GROQ_API_KEY` | Groq API key | — |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OPENAI_MODEL` | OpenAI chat model | `gpt-4o-mini` |
| `EVAL_SAMPLE_SIZE` | Samples for evaluation | `100` |

---

## Execution

### Interactive CLI

```bash
python main.py
```

### Non-Interactive Demos

```bash
# Demo all prompt techniques
python main.py --demo prompts --provider openai

# Sentiment analysis workflow
python main.py --demo sentiment --provider openai

# Spam detection workflow
python main.py --demo spam --provider groq

# Text-to-image generation
python main.py --demo text2img

# Image-to-image transformation
python main.py --demo img2img

# Use Ollama (local)
python main.py --demo prompts --provider ollama
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/prompt_engineering_analysis.ipynb
```

### Run Tests

```bash
pytest tests/ -v
```

### Generate Documentation Artifacts

```bash
# PowerPoint presentation (15-20 slides)
python docs/generate_presentation.py

# PDF academic report
python docs/generate_report.py
```

---

## Prompt Engineering Techniques

| Technique | File | Description |
|-----------|------|-------------|
| Simple Prompt | `prompts/simple_prompt.py` | Direct instruction, no context |
| Zero-Shot | `prompts/zero_shot_prompt.py` | Task description without examples |
| Few-Shot | `prompts/few_shot_prompt.py` | 2-5 labeled examples before input |
| Chain of Thought | `prompts/chain_of_thought_prompt.py` | Step-by-step reasoning |
| Self-Consistency | `prompts/self_consistency_prompt.py` | Multiple samples + majority vote |
| Role Prompt | `prompts/role_prompt.py` | Expert persona via system message |
| Step-Back | `prompts/step_back_prompt.py` | Derive principles, then apply |
| Tree of Thoughts | `prompts/tree_of_thoughts_prompt.py` | Multi-branch reasoning (bonus) |

---

## Examples

### Quick Sentiment Classification

```python
from llms.openai_client import OpenAIClient
from prompts.zero_shot_prompt import ZeroShotPrompt, SENTIMENT_ZERO_SHOT

client = OpenAIClient()
response = ZeroShotPrompt.run(
    client,
    SENTIMENT_ZERO_SHOT,
    "This movie was absolutely brilliant!"
)
print(response.content)  # positive
```

### Full Evaluation Pipeline

```python
from use_cases.sentiment_analysis import SentimentAnalysisUseCase
from llms.openai_client import OpenAIClient

client = OpenAIClient()
use_case = SentimentAnalysisUseCase(client)
df = use_case.load_and_prepare(sample_size=20)
comparison = use_case.run_all_techniques(df)
print(comparison)
```

---

## Output Artifacts

After running experiments, check the `outputs/` directory:

| File | Description |
|------|-------------|
| `sentiment_comparison.csv` | Technique comparison metrics |
| `technique_comparison.png` | Accuracy/speed/cost bar charts |
| `imdb_eda.png` | Sentiment dataset EDA |
| `spam_eda.png` | Spam dataset EDA |
| `cm_*.png` | Confusion matrices per technique |
| `Prompt_Engineering_Presentation.pptx` | Generated slides |
| `Prompt_Engineering_Report.pdf` | Academic report |

---

## Screenshots

> Place screenshots of CLI output, charts, and generated images in `images/screenshots/` after running the project.

Expected outputs:
- Rich console tables with technique comparison
- Matplotlib bar charts comparing accuracy, speed, cost
- DALL-E generated images in `images/generated/`

---

## LLM Provider Comparison

| Provider | Models | Cost | Speed | Local |
|----------|--------|------|-------|-------|
| OpenAI | GPT-4o-mini, DALL-E 3 | Pay-per-token | Medium | No |
| Ollama | Llama3, Mistral, Gemma | Free | Depends on HW | Yes |
| Groq | Llama 3.3 70B, Mixtral | Pay-per-token | Very Fast | No |

---

## License

Academic project for Master BDCC program. For educational purposes.

---

## Author

Master BDCC Student — Université Mohammed VI Polytechnique — 2026
