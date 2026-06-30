# Architecture Documentation

## Prompt Engineering with Python using LLMs

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                           │
│                    Interactive Menu / Demos                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  Use Cases  │  │   Prompts   │  │  Evaluation │
   │             │  │             │  │             │
   │ • Sentiment │  │ • Simple    │  │ • Metrics   │
   │ • Spam      │  │ • Zero-Shot │  │ • Evaluator │
   │ • Text2Img  │  │ • Few-Shot  │  │ • Cost Est. │
   │ • Img2Img   │  │ • CoT       │  │             │
   └──────┬──────┘  │ • Role      │  └──────┬──────┘
          │         │ • Self-Cons.│         │
          │         │ • Step-Back │         │
          │         │ • ToT       │         │
          │         └──────┬──────┘         │
          │                │                │
          └────────────────┼────────────────┘
                           ▼
              ┌────────────────────────┐
              │     LLM Clients        │
              │  (BaseLLMClient ABC)   │
              ├──────────┬─────────────┤
              │ OpenAI   │ Ollama │ Groq│
              └──────────┴─────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │     Configuration      │
              │  settings.py + .env    │
              └────────────────────────┘
```

### Layer Description

| Layer | Responsibility |
|-------|---------------|
| **Presentation** | `main.py` CLI, Jupyter notebooks, Rich console output |
| **Use Cases** | Business logic orchestrating datasets, prompts, and evaluation |
| **Prompts** | Prompt engineering technique implementations |
| **LLMs** | Provider-specific API clients with retry and metrics |
| **Evaluation** | Metrics computation, comparison, cost estimation |
| **Datasets** | Data loading, cleaning, visualization |
| **Config** | Environment-based settings, model registry |
| **Utils** | Logging, console formatting |

### Design Patterns

- **Abstract Factory**: `get_client(provider)` creates the appropriate LLM client
- **Strategy Pattern**: Each prompt technique is interchangeable via common interface
- **Template Method**: `BaseLLMClient.chat()` defines retry skeleton, subclasses implement `_chat_completion`
- **Singleton**: `get_settings()` returns cached configuration

### Data Flow

1. User selects a use case via CLI
2. Dataset is loaded and preprocessed
3. For each prompting technique, predictions are generated via LLM client
4. Predictions are compared against gold labels
5. Metrics, confusion matrices, and comparison charts are generated
6. Results saved to `outputs/`
